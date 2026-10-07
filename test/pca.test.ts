import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
  fitPca, validatePoints, projectPoint, reconstructPoint, verifyModel,
  explainedVarianceRatios, PcaError, Point,
} from "../src/pca";
import { openDb, saveDataset, fitAndSaveModel, loadModel, loadDatasetPoints } from "../src/db";
import { buildProjectionReport } from "../src/report";

const approx = (a: number, b: number, tol = 1e-9) =>
  assert.ok(Math.abs(a - b) <= tol, `${a} ≉ ${b}`);

// 已知例子：(-1,0),(0,0),(1,0)
test("水平三点：均值原点、协方差 diag(1,0)、第一主轴沿 x、k=1 无损", () => {
  const pts: Point[] = [
    { id: "a", x: -1, y: 0 },
    { id: "b", x: 0, y: 0 },
    { id: "c", x: 1, y: 0 },
  ];
  const m = fitPca(pts);
  approx(m.mean.x, 0); approx(m.mean.y, 0);
  approx(m.cov.c11, 1); approx(m.cov.c12, 0); approx(m.cov.c22, 0);
  approx(m.eigenvalues[0], 1); approx(m.eigenvalues[1], 0);
  approx(Math.abs(m.axes[0].x), 1); approx(m.axes[0].y, 0);
  assert.ok(m.axes[0].x > 0, "符号规则：第一分量取正");
  const r = buildProjectionReport(m, pts, 1);
  approx(r.totalSqError, 0, 1e-12);
  const { ratios } = explainedVarianceRatios(m);
  approx(ratios[0]!, 1); approx(ratios[1]!, 0);
});

test("整体平移：均值平移，协方差/主轴/特征值不变", () => {
  const base: Point[] = [
    { id: "1", x: 0, y: 0 }, { id: "2", x: 1, y: 2 }, { id: "3", x: 3, y: 1 }, { id: "4", x: 2, y: 5 },
  ];
  const shifted = base.map((p) => ({ id: p.id, x: p.x + 100, y: p.y - 40 }));
  const m1 = fitPca(base);
  const m2 = fitPca(shifted);
  approx(m2.mean.x, m1.mean.x + 100); approx(m2.mean.y, m1.mean.y - 40);
  approx(m2.cov.c11, m1.cov.c11); approx(m2.cov.c12, m1.cov.c12); approx(m2.cov.c22, m1.cov.c22);
  approx(m2.eigenvalues[0], m1.eigenvalues[0]); approx(m2.eigenvalues[1], m1.eigenvalues[1]);
  approx(m2.axes[0].x, m1.axes[0].x); approx(m2.axes[0].y, m1.axes[0].y);
});

test("点顺序变化：模型完全一致", () => {
  const pts: Point[] = [
    { id: "1", x: 0, y: 0 }, { id: "2", x: 1, y: 2 }, { id: "3", x: 3, y: 1 }, { id: "4", x: 2, y: 5 },
  ];
  const shuffled = [pts[2], pts[0], pts[3], pts[1]];
  const m1 = fitPca(pts);
  const m2 = fitPca(shuffled);
  approx(m2.cov.c11, m1.cov.c11); approx(m2.cov.c12, m1.cov.c12);
  approx(m2.eigenvalues[0], m1.eigenvalues[0]);
  approx(m2.axes[0].x, m1.axes[0].x); approx(m2.axes[0].y, m1.axes[0].y);
});

test("全体相同点：零方差，固定坐标轴，比例无 NaN", () => {
  const pts: Point[] = [
    { id: "1", x: 5, y: 5 }, { id: "2", x: 5, y: 5 }, { id: "3", x: 5, y: 5 },
  ];
  const m = fitPca(pts);
  approx(m.eigenvalues[0], 0); approx(m.eigenvalues[1], 0);
  assert.deepEqual(m.axes[0], { x: 1, y: 0 });
  assert.deepEqual(m.axes[1], { x: 0, y: 1 });
  assert.ok(m.notes.some((n) => n.includes("zero_total_variance")));
  const { ratios, note } = explainedVarianceRatios(m);
  assert.deepEqual(ratios, [null, null]);
  assert.ok(note!.includes("zero_total_variance"));
  const r = buildProjectionReport(m, pts, 1);
  approx(r.totalSqError, 0, 1e-12);
});

test("各向同性四点：重复特征值，主轴不唯一说明", () => {
  const pts: Point[] = [
    { id: "1", x: 1, y: 0 }, { id: "2", x: -1, y: 0 }, { id: "3", x: 0, y: 1 }, { id: "4", x: 0, y: -1 },
  ];
  const m = fitPca(pts);
  approx(m.eigenvalues[0], m.eigenvalues[1], 1e-12);
  assert.ok(m.notes.some((n) => n.includes("equal_eigenvalues")));
});

test("斜线点：主轴旋转 45°", () => {
  const pts: Point[] = [
    { id: "1", x: 0, y: 0 }, { id: "2", x: 1, y: 1 }, { id: "3", x: 2, y: 2 }, { id: "4", x: 3, y: 3 },
  ];
  const m = fitPca(pts);
  approx(m.axes[0].x, Math.SQRT1_2, 1e-9);
  approx(m.axes[0].y, Math.SQRT1_2, 1e-9);
  approx(m.eigenvalues[1], 0, 1e-12);
});

test("不同尺度：缩放影响主轴（不自动标准化）", () => {
  const pts: Point[] = [
    { id: "1", x: 0, y: 0 }, { id: "2", x: 1, y: 1 }, { id: "3", x: 2, y: 0 }, { id: "4", x: 3, y: 1 },
  ];
  const scaled = pts.map((p) => ({ ...p, y: p.y * 1000 }));
  const m1 = fitPca(pts);
  const m2 = fitPca(scaled);
  assert.ok(Math.abs(m1.axes[0].y) > 0.1);
  assert.ok(Math.abs(m2.axes[0].y) > 0.99, "y 放大后主轴几乎沿 y");
});

test("近共线：λ2 很小但非负，k=1 误差等于 (n-1)*λ2", () => {
  const pts: Point[] = [
    { id: "1", x: 0, y: 0 }, { id: "2", x: 1, y: 1e-4 }, { id: "3", x: 2, y: 0 }, { id: "4", x: 3, y: 1e-4 },
  ];
  const m = fitPca(pts);
  assert.ok(m.eigenvalues[1] >= 0);
  assert.ok(m.eigenvalues[1] < 1e-6);
  const r = buildProjectionReport(m, pts, 1);
  approx(r.totalSqError, (pts.length - 1) * m.eigenvalues[1], 1e-12);
});

test("k=2 重建在容差内恢复所有点；k=1 训练总平方误差 = (n-1)*λ2", () => {
  const pts: Point[] = [
    { id: "1", x: 0.5, y: -1.2 }, { id: "2", x: 3, y: 4 }, { id: "3", x: -2, y: 0.3 },
    { id: "4", x: 7, y: 7 }, { id: "5", x: -4, y: -4 },
  ];
  const m = fitPca(pts);
  for (const p of pts) {
    const rec = reconstructPoint(m, projectPoint(m, p, 2));
    approx(rec.x, p.x, 1e-9); approx(rec.y, p.y, 1e-9);
  }
  const r = buildProjectionReport(m, pts, 1);
  approx(r.totalSqError, (pts.length - 1) * m.eigenvalues[1], 1e-9);
});

test("投影用训练均值：新点不参与拟合", () => {
  const train: Point[] = [
    { id: "1", x: 0, y: 0 }, { id: "2", x: 2, y: 1 }, { id: "3", x: 4, y: 2 }, { id: "4", x: 6, y: 3 },
  ];
  const m = fitPca(train);
  const before = JSON.stringify(m);
  const s = projectPoint(m, { x: 100, y: 50 }, 2);
  approx(s[0], (100 - m.mean.x) * m.axes[0].x + (50 - m.mean.y) * m.axes[0].y);
  assert.equal(JSON.stringify(m), before, "投影不得修改模型");
});

test("核验入口：正交归一、C·v=λ·v、重建误差", () => {
  const pts: Point[] = [
    { id: "1", x: 1, y: 2 }, { id: "2", x: 4, y: 1 }, { id: "3", x: -3, y: 5 }, { id: "4", x: 0, y: -2 },
  ];
  const m = fitPca(pts);
  const v = verifyModel(m, pts);
  assert.ok(v.ok, JSON.stringify(v.checks));
});

test("明显负特征值报错", () => {
  const { eigenDecompose } = require("../src/pca") as typeof import("../src/pca");
  assert.throws(() => eigenDecompose({ c11: -5, c12: 0, c22: 1 }), (e: PcaError) => e.code === "NEGATIVE_EIGENVALUE");
});

test("输入校验：缺失/非有限/过少/超范围/重复编号", () => {
  assert.throws(() => validatePoints([{ id: "1", x: 0, y: 0 }]), (e: PcaError) => e.code === "TOO_FEW_POINTS");
  assert.throws(() => validatePoints([{ id: "1", x: 0, y: NaN }, { id: "2", x: 1, y: 1 }]), (e: PcaError) => e.code === "NON_FINITE_VALUE");
  assert.throws(() => validatePoints([{ id: "1", x: 0 }, { id: "2", x: 1, y: 1 }]), (e: PcaError) => e.code === "NON_FINITE_VALUE");
  assert.throws(() => validatePoints([{ id: "1", x: 1e13, y: 0 }, { id: "2", x: 1, y: 1 }]), (e: PcaError) => e.code === "COORD_OUT_OF_RANGE");
  assert.throws(() => validatePoints([{ id: "1", x: 0, y: 0 }, { id: "1", x: 1, y: 1 }]), (e: PcaError) => e.code === "DUPLICATE_ID");
});

function tmpDb(): string {
  return path.join(fs.mkdtempSync(path.join(os.tmpdir(), "pca2d-")), "test.db");
}

test("存储：保存-重启-复现；模型引用明确数据集版本", () => {
  const dbPath = tmpDb();
  const pts = [
    { id: "a", x: -1, y: 0 }, { id: "b", x: 0, y: 0 }, { id: "c", x: 1, y: 0 },
  ];
  let modelId: number;
  {
    const db = openDb(dbPath);
    const { datasetId } = saveDataset(db, "t", pts);
    modelId = fitAndSaveModel(db, datasetId).modelId;
    db.close();
  }
  {
    const db = openDb(dbPath); // 模拟重启
    const { model, datasetId } = loadModel(db, modelId);
    approx(model.mean.x, 0); approx(model.cov.c11, 1);
    const loaded = loadDatasetPoints(db, datasetId);
    assert.equal(loaded.length, 3);
    assert.ok(verifyModel(model, loaded).ok);
    db.close();
  }
});

test("失败回滚：非法输入不产生数据集/模型，不污染已有模型", () => {
  const dbPath = tmpDb();
  const db = openDb(dbPath);
  const good = saveDataset(db, "good", [{ id: "a", x: 0, y: 0 }, { id: "b", x: 1, y: 1 }]);
  const goodModel = fitAndSaveModel(db, good.datasetId);
  assert.throws(() => saveDataset(db, "bad", [{ id: "x", x: NaN, y: 0 }, { id: "y", x: 1, y: 1 }]));
  assert.throws(() => fitAndSaveModel(db, 9999), (e: PcaError) => e.code === "DATASET_NOT_FOUND");
  const datasets = db.prepare("SELECT COUNT(*) c FROM datasets").get() as { c: number };
  const models = db.prepare("SELECT COUNT(*) c FROM models").get() as { c: number };
  assert.equal(datasets.c, 1);
  assert.equal(models.c, 1);
  const { model } = loadModel(db, goodModel.modelId);
  approx(model.mean.x, 0.5);
  db.close();
});

test("修改样本产生新模型版本", () => {
  const dbPath = tmpDb();
  const db = openDb(dbPath);
  const d1 = saveDataset(db, "v1", [{ id: "a", x: 0, y: 0 }, { id: "b", x: 1, y: 0 }]);
  const d2 = saveDataset(db, "v2", [{ id: "a", x: 0, y: 0 }, { id: "b", x: 2, y: 0 }]);
  const m1 = fitAndSaveModel(db, d1.datasetId);
  const m2 = fitAndSaveModel(db, d2.datasetId);
  assert.notEqual(m1.modelId, m2.modelId);
  approx(loadModel(db, m1.modelId).model.cov.c11, 0.5);
  approx(loadModel(db, m2.modelId).model.cov.c11, 2);
  db.close();
});
