// SQLite 持久化：数据集版本、模型、投影结果。失败时事务回滚，不残留半成品。
import { DatabaseSync } from "node:sqlite";
import * as fs from "node:fs";
import * as path from "node:path";
import {
  DEFAULT_TOLERANCE,
  PcaError,
  PcaModel,
  Point,
  fitPca,
  validatePoints,
} from "./pca";

export const DATA_DIR = path.join(process.cwd(), "data");
export const DB_PATH = path.join(DATA_DIR, "pca2d.db");

export function openDb(dbPath: string = DB_PATH): DatabaseSync {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new DatabaseSync(dbPath);
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;
    CREATE TABLE IF NOT EXISTS datasets (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS dataset_points (
      dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
      point_id TEXT NOT NULL,
      x REAL NOT NULL,
      y REAL NOT NULL,
      PRIMARY KEY (dataset_id, point_id)
    );
    CREATE TABLE IF NOT EXISTS models (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
      n INTEGER NOT NULL,
      mean_x REAL NOT NULL, mean_y REAL NOT NULL,
      cov_c11 REAL NOT NULL, cov_c12 REAL NOT NULL, cov_c22 REAL NOT NULL,
      lambda1 REAL NOT NULL, lambda2 REAL NOT NULL,
      axis1_x REAL NOT NULL, axis1_y REAL NOT NULL,
      axis2_x REAL NOT NULL, axis2_y REAL NOT NULL,
      tolerance REAL NOT NULL,
      notes TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS projections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      k INTEGER NOT NULL,
      point_id TEXT NOT NULL,
      x REAL NOT NULL, y REAL NOT NULL,
      score1 REAL NOT NULL, score2 REAL,
      rec_x REAL NOT NULL, rec_y REAL NOT NULL,
      sq_error REAL NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);
  return db;
}

export function saveDataset(db: DatabaseSync, name: string, rawPoints: unknown): { datasetId: number; points: Point[] } {
  const points = validatePoints(rawPoints); // 校验先于任何写库
  try {
    db.exec("BEGIN");
    const res = db.prepare("INSERT INTO datasets (name) VALUES (?)").run(String(name ?? "dataset"));
    const datasetId = Number(res.lastInsertRowid);
    const ins = db.prepare("INSERT INTO dataset_points (dataset_id, point_id, x, y) VALUES (?, ?, ?, ?)");
    for (const p of points) ins.run(datasetId, p.id, p.x, p.y);
    db.exec("COMMIT");
    return { datasetId, points };
  } catch (e) {
    db.exec("ROLLBACK");
    throw new PcaError("STORAGE_FAILED", `保存样本失败，已回滚: ${(e as Error).message}`);
  }
}

export function loadDatasetPoints(db: DatabaseSync, datasetId: number): Point[] {
  const rows = db
    .prepare("SELECT point_id, x, y FROM dataset_points WHERE dataset_id = ? ORDER BY rowid")
    .all(datasetId) as unknown as { point_id: string; x: number; y: number }[];
  if (rows.length === 0) throw new PcaError("DATASET_NOT_FOUND", `数据集 ${datasetId} 不存在或为空`);
  return rows.map((r) => ({ id: r.point_id, x: r.x, y: r.y }));
}

export function listDatasets(db: DatabaseSync) {
  return db
    .prepare(
      `SELECT d.id, d.name, d.created_at, COUNT(p.point_id) AS n
       FROM datasets d LEFT JOIN dataset_points p ON p.dataset_id = d.id
       GROUP BY d.id ORDER BY d.id`,
    )
    .all();
}

export function fitAndSaveModel(
  db: DatabaseSync,
  datasetId: number,
  tolerance: number = DEFAULT_TOLERANCE,
): { modelId: number; model: PcaModel; points: Point[] } {
  const points = loadDatasetPoints(db, datasetId);
  const model = fitPca(points, tolerance);
  try {
    db.exec("BEGIN");
    const res = db
      .prepare(
        `INSERT INTO models (dataset_id, n, mean_x, mean_y, cov_c11, cov_c12, cov_c22,
          lambda1, lambda2, axis1_x, axis1_y, axis2_x, axis2_y, tolerance, notes)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        datasetId, model.n, model.mean.x, model.mean.y,
        model.cov.c11, model.cov.c12, model.cov.c22,
        model.eigenvalues[0], model.eigenvalues[1],
        model.axes[0].x, model.axes[0].y, model.axes[1].x, model.axes[1].y,
        model.tolerance, JSON.stringify(model.notes),
      );
    db.exec("COMMIT");
    return { modelId: Number(res.lastInsertRowid), model, points };
  } catch (e) {
    db.exec("ROLLBACK");
    throw new PcaError("STORAGE_FAILED", `保存模型失败，已回滚: ${(e as Error).message}`);
  }
}

export function loadModel(db: DatabaseSync, modelId: number): { model: PcaModel; datasetId: number } {
  const row = db.prepare("SELECT * FROM models WHERE id = ?").get(modelId) as Record<string, unknown> | undefined;
  if (!row) throw new PcaError("MODEL_NOT_FOUND", `模型 ${modelId} 不存在`);
  const model: PcaModel = {
    n: row.n as number,
    mean: { x: row.mean_x as number, y: row.mean_y as number },
    cov: { c11: row.cov_c11 as number, c12: row.cov_c12 as number, c22: row.cov_c22 as number },
    eigenvalues: [row.lambda1 as number, row.lambda2 as number],
    axes: [
      { x: row.axis1_x as number, y: row.axis1_y as number },
      { x: row.axis2_x as number, y: row.axis2_y as number },
    ],
    tolerance: row.tolerance as number,
    notes: JSON.parse(row.notes as string) as string[],
  };
  return { model, datasetId: row.dataset_id as number };
}

export function listModels(db: DatabaseSync) {
  return db.prepare("SELECT id, dataset_id, n, lambda1, lambda2, created_at FROM models ORDER BY id").all();
}

export function saveProjection(
  db: DatabaseSync,
  modelId: number,
  k: number,
  rows: { id: string; x: number; y: number; scores: number[]; rec: { x: number; y: number }; sqError: number }[],
): void {
  try {
    db.exec("BEGIN");
    const ins = db.prepare(
      `INSERT INTO projections (model_id, k, point_id, x, y, score1, score2, rec_x, rec_y, sq_error)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    );
    for (const r of rows) {
      ins.run(modelId, k, r.id, r.x, r.y, r.scores[0], r.scores.length > 1 ? r.scores[1] : null,
        r.rec.x, r.rec.y, r.sqError);
    }
    db.exec("COMMIT");
  } catch (e) {
    db.exec("ROLLBACK");
    throw new PcaError("STORAGE_FAILED", `保存投影结果失败，已回滚: ${(e as Error).message}`);
  }
}

export function listProjections(db: DatabaseSync, modelId: number) {
  return db
    .prepare("SELECT * FROM projections WHERE model_id = ? ORDER BY id")
    .all(modelId);
}
