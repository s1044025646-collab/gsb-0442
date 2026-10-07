/** SQLite 持久化：样本版本、模型、投影结果。所有多步写入都在事务内，失败整体回滚。 */
import { DatabaseSync } from "node:sqlite";
import * as fs from "node:fs";
import * as path from "node:path";
import { FitResult, InputPoint, ModelLike, PcaError, ProjectionRow } from "./pca";

export interface SampleRow { id: number; name: string; n: number; created_at: string }
export interface ModelRow extends ModelLike {
  id: number; sample_id: number; total_variance: number;
  ratio1: number; ratio2: number; degenerate: string; notes: string[]; created_at: string;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  n INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS sample_points (
  sample_id INTEGER NOT NULL REFERENCES samples(id),
  idx INTEGER NOT NULL,
  point_id TEXT NOT NULL,
  x REAL NOT NULL,
  y REAL NOT NULL,
  PRIMARY KEY (sample_id, idx)
);
CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sample_id INTEGER NOT NULL REFERENCES samples(id),
  mean_x REAL NOT NULL, mean_y REAL NOT NULL,
  cov_a REAL NOT NULL, cov_b REAL NOT NULL, cov_c REAL NOT NULL,
  lambda1 REAL NOT NULL, lambda2 REAL NOT NULL,
  a1x REAL NOT NULL, a1y REAL NOT NULL, a2x REAL NOT NULL, a2y REAL NOT NULL,
  tol REAL NOT NULL, total_variance REAL NOT NULL,
  ratio1 REAL NOT NULL, ratio2 REAL NOT NULL,
  degenerate TEXT NOT NULL, notes TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS projections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(id),
  k INTEGER NOT NULL,
  point_id TEXT NOT NULL, x REAL NOT NULL, y REAL NOT NULL,
  score1 REAL NOT NULL, score2 REAL NOT NULL,
  rec_x REAL NOT NULL, rec_y REAL NOT NULL, err2 REAL NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);`;

export class Store {
  readonly db: DatabaseSync;
  constructor(dbPath: string) {
    if (dbPath !== ":memory:") fs.mkdirSync(path.dirname(dbPath), { recursive: true });
    this.db = new DatabaseSync(dbPath);
    this.db.exec("PRAGMA journal_mode = WAL;");
    this.db.exec("PRAGMA foreign_keys = ON;");
    this.db.exec(SCHEMA);
  }
  close(): void { this.db.close(); }

  /** 保存样本新版本；任一点写入失败则整体回滚，不留半截样本。 */
  saveSample(name: string, points: InputPoint[]): number {
    if (typeof name !== "string" || name.trim().length === 0) throw new PcaError("INVALID_INPUT", "样本名不能为空");
    try {
      this.db.exec("BEGIN");
      const info = this.db.prepare("INSERT INTO samples (name, n) VALUES (?, ?)").run(name, points.length);
      const sampleId = Number(info.lastInsertRowid);
      const stmt = this.db.prepare("INSERT INTO sample_points (sample_id, idx, point_id, x, y) VALUES (?, ?, ?, ?, ?)");
      points.forEach((p, i) => stmt.run(sampleId, i, p.id, p.x, p.y));
      this.db.exec("COMMIT");
      return sampleId;
    } catch (err) {
      try { this.db.exec("ROLLBACK"); } catch { /* 事务可能未开启 */ }
      throw new PcaError("DB_ERROR", `保存样本失败，已回滚: ${(err as Error).message}`);
    }
  }

  getSamplePoints(sampleId: number): InputPoint[] {
    const rows = this.db.prepare("SELECT point_id, x, y FROM sample_points WHERE sample_id = ? ORDER BY idx").all(sampleId) as unknown as { point_id: string; x: number; y: number }[];
    if (rows.length === 0) throw new PcaError("NOT_FOUND", `样本 ${sampleId} 不存在或无点`);
    return rows.map((r) => ({ id: r.point_id, x: r.x, y: r.y }));
  }

  listSamples(): SampleRow[] {
    return this.db.prepare("SELECT id, name, n, created_at FROM samples ORDER BY id").all() as unknown as SampleRow[];
  }

  saveModel(sampleId: number, fit: FitResult): number {
    try {
      this.db.exec("BEGIN");
      const info = this.db.prepare(`INSERT INTO models
        (sample_id, mean_x, mean_y, cov_a, cov_b, cov_c, lambda1, lambda2, a1x, a1y, a2x, a2y, tol, total_variance, ratio1, ratio2, degenerate, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`).run(
        sampleId, fit.mean.x, fit.mean.y, fit.cov.a, fit.cov.b, fit.cov.c,
        fit.lambda1, fit.lambda2, fit.axis1.x, fit.axis1.y, fit.axis2.x, fit.axis2.y,
        fit.tol, fit.totalVariance, fit.explainedRatio[0], fit.explainedRatio[1],
        fit.degenerate, JSON.stringify(fit.notes));
      this.db.exec("COMMIT");
      return Number(info.lastInsertRowid);
    } catch (err) {
      try { this.db.exec("ROLLBACK"); } catch { /* 事务可能未开启 */ }
      throw new PcaError("DB_ERROR", `保存模型失败，已回滚: ${(err as Error).message}`);
    }
  }

  getModel(modelId: number): ModelRow {
    const r = this.db.prepare("SELECT * FROM models WHERE id = ?").get(modelId) as Record<string, unknown> | undefined;
    if (!r) throw new PcaError("NOT_FOUND", `模型 ${modelId} 不存在`);
    return {
      id: r.id as number, sample_id: r.sample_id as number,
      mean: { x: r.mean_x as number, y: r.mean_y as number },
      cov: { a: r.cov_a as number, b: r.cov_b as number, c: r.cov_c as number },
      lambda1: r.lambda1 as number, lambda2: r.lambda2 as number,
      axis1: { x: r.a1x as number, y: r.a1y as number },
      axis2: { x: r.a2x as number, y: r.a2y as number },
      tol: r.tol as number, total_variance: r.total_variance as number,
      ratio1: r.ratio1 as number, ratio2: r.ratio2 as number,
      degenerate: r.degenerate as string, notes: JSON.parse(r.notes as string) as string[],
      created_at: r.created_at as string,
    };
  }

  listModels(): { id: number; sample_id: number; lambda1: number; lambda2: number; degenerate: string; created_at: string }[] {
    return this.db.prepare("SELECT id, sample_id, lambda1, lambda2, degenerate, created_at FROM models ORDER BY id").all() as never;
  }

  saveProjections(modelId: number, k: number, rows: ProjectionRow[]): void {
    try {
      this.db.exec("BEGIN");
      const stmt = this.db.prepare(`INSERT INTO projections (model_id, k, point_id, x, y, score1, score2, rec_x, rec_y, err2)
        VALUES (?,?,?,?,?,?,?,?,?,?)`);
      for (const r of rows) stmt.run(modelId, k, r.id, r.x, r.y, r.score1, r.score2, r.recX, r.recY, r.err2);
      this.db.exec("COMMIT");
    } catch (err) {
      try { this.db.exec("ROLLBACK"); } catch { /* 事务可能未开启 */ }
      throw new PcaError("DB_ERROR", `保存投影结果失败，已回滚: ${(err as Error).message}`);
    }
  }

  listProjections(modelId: number): Record<string, unknown>[] {
    return this.db.prepare("SELECT * FROM projections WHERE model_id = ? ORDER BY id").all(modelId) as never;
  }
}

export function defaultDbPath(): string {
  return process.env.PCA_DB ?? path.join(process.cwd(), "data", "pca.db");
}



