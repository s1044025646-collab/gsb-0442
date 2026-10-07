import { test } from "node:test";
import assert from "node:assert/strict";
import { fitPca, projectPoints, verifyModel, covariance2, mean2, InputPoint, PcaError } from "../src/pca";

const closeTo = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

test("已知例 (-1,0),(0,0),(1,0)", () => {
  const pts = [{ id: "a", x: -1, y: 0 }, { id: "b", x: 0, y: 0 }, { id: "c", x: 1, y: 0 }];
  const fit = fitPca(pts);
  assert.ok(closeTo(fit.mean.x, 0) && closeTo(fit.mean.y, 0));
  assert.ok(closeTo(fit.cov.a, 1) && closeTo(fit.cov.b, 0) && closeTo(fit.cov.c, 0));
  assert.ok(closeTo(fit.lambda1, 1) && closeTo(fit.lambda2, 0));
  assert.ok(closeTo(Math.abs(fit.axis1.x), 1) && closeTo(fit.axis1.y, 0));
  const rows = projectPoints(fit, pts, 1);
  for (const [i, r] of rows.entries()) {
    assert.ok(closeTo(r.err2, 0, 1e-12), "保留一维无损");
    assert.ok(closeTo(r.recX, pts[i].x) && closeTo(r.recY, pts[i].y));
  }
});

test("协方差用 n-1 且先中心化（与手算外积核对）", () => {
  const pts: InputPoint[] = [{ id: "1", x: 2, y: 5 }, { id: "2", x: -3, y: 1 }, { id: "3", x: 4, y: -2 }, { id: "4", x: 0, y: 7 }];
  const m = mean2(pts);
  const c = covariance2(pts, m);
  // 独立手算
  let a = 0, b = 0, cc = 0;
  for (const p of pts) { const dx = p.x - m.x, dy = p.y - m.y; a += dx * dx; b += dx * dy; cc += dy * dy; }
  assert.ok(closeTo(c.a, a / 3) && closeTo(c.b, b / 3) && closeTo(c.c, cc / 3));
  // 与未去均值二阶矩不同
  const m2 = pts.reduce((s, p) => s + p.x * p.x, 0) / 3;
  assert.ok(!closeTo(c.a, m2));
});

test("整体平移不改变协方差与主轴", () => {
  const base: InputPoint[] = [{ id: "1", x: 0, y: 0 }, { id: "2", x: 1, y: 2 }, { id: "3", x: 3, y: 1 }, { id: "4", x: -1, y: 4 }];
  const shifted = base.map((p) => ({ id: p.id, x: p.x + 100, y: p.y - 50 }));
  const f1 = fitPca(base), f2 = fitPca(shifted);
  assert.ok(closeTo(f2.mean.x, f1.mean.x + 100) && closeTo(f2.mean.y, f1.mean.y - 50));
  assert.ok(closeTo(f1.cov.a, f2.cov.a) && closeTo(f1.cov.b, f2.cov.b) && closeTo(f1.cov.c, f2.cov.c));
  assert.ok(closeTo(f1.axis1.x, f2.axis1.x) && closeTo(f1.axis1.y, f2.axis1.y));
  assert.ok(closeTo(f1.lambda1, f2.lambda1) && closeTo(f1.lambda2, f2.lambda2));
});

test("点顺序变化结果一致", () => {
  const pts: InputPoint[] = [{ id: "1", x: 0.5, y: -1 }, { id: "2", x: 2, y: 2 }, { id: "3", x: -3, y: 0.25 }, { id: "4", x: 1, y: 1 }, { id: "5", x: 4, y: -2 }];
  const rev = [...pts].reverse();
  const f1 = fitPca(pts), f2 = fitPca(rev);
  assert.ok(closeTo(f1.mean.x, f2.mean.x) && closeTo(f1.lambda1, f2.lambda1) && closeTo(f1.lambda2, f2.lambda2));
  assert.ok(closeTo(f1.axis1.x, f2.axis1.x) && closeTo(f1.axis1.y, f2.axis1.y));
});

test("全重合点：零方差、固定坐标轴、无 NaN", () => {
  const pts = Array.from({ length: 5 }, (_, i) => ({ id: String(i), x: 3, y: -2 }));
  const fit = fitPca(pts);
  assert.equal(fit.degenerate, "zero-variance");
  assert.deepEqual(fit.axis1, { x: 1, y: 0 });
  assert.deepEqual(fit.axis2, { x: 0, y: 1 });
  assert.deepEqual(fit.explainedRatio, [0, 0]);
  for (const v of [fit.lambda1, fit.lambda2, fit.explainedRatio[0], fit.explainedRatio[1]]) assert.ok(Number.isFinite(v));
  const rows = projectPoints(fit, pts, 1);
  assert.ok(rows.every((r) => r.err2 === 0));
});

test("各向同性四点：重复特征值，主轴不唯一用确定基", () => {
  const pts = [{ id: "1", x: 1, y: 1 }, { id: "2", x: 1, y: -1 }, { id: "3", x: -1, y: 1 }, { id: "4", x: -1, y: -1 }];
  const fit = fitPca(pts);
  assert.equal(fit.degenerate, "repeated-eigenvalue");
  assert.ok(closeTo(fit.lambda1, fit.lambda2));
  assert.ok(fit.notes.some((n) => n.includes("不唯一")));
});

test("近共线点：λ2 很小但非负，核验通过", () => {
  const pts: InputPoint[] = Array.from({ length: 20 }, (_, i) => ({ id: String(i), x: i, y: 2 * i + 1e-9 * (i % 3) }));
  const fit = fitPca(pts);
  assert.ok(fit.lambda2 >= 0);
  assert.ok(fit.lambda2 / fit.lambda1 < 1e-12);
  assert.ok(verifyModel(fit, pts).ok);
});

test("不同尺度影响主轴（不缩放）", () => {
  const pts: InputPoint[] = [{ id: "1", x: 0, y: 0 }, { id: "2", x: 1000, y: 1 }, { id: "3", x: -1000, y: -1 }, { id: "4", x: 500, y: -2 }];
  const fit = fitPca(pts);
  assert.ok(Math.abs(fit.axis1.x) > 0.999, "大尺度坐标主导主轴");
  assert.ok(fit.notes.some((n) => n.includes("未按坐标标准差缩放")));
});

test("保留两维在容差内恢复所有点；保留一维 SSE=(n-1)λ2", () => {
  const pts: InputPoint[] = Array.from({ length: 30 }, (_, i) => ({ id: String(i), x: Math.sin(i) * 7 + i, y: Math.cos(i * 1.7) * 3 - 0.5 * i }));
  const fit = fitPca(pts);
  const rows2 = projectPoints(fit, pts, 2);
  for (const [i, r] of rows2.entries()) {
    assert.ok(closeTo(r.recX, pts[i].x, 1e-9) && closeTo(r.recY, pts[i].y, 1e-9));
  }
  const sse1 = projectPoints(fit, pts, 1).reduce((s, r) => s + r.err2, 0);
  assert.ok(closeTo(sse1, (pts.length - 1) * fit.lambda2, 1e-9));
  assert.ok(verifyModel(fit, pts).ok);
});

test("投影得分与独立内积核对一致", () => {
  const train: InputPoint[] = [{ id: "1", x: 1, y: 2 }, { id: "2", x: 3, y: -1 }, { id: "3", x: -2, y: 0.5 }, { id: "4", x: 0, y: 4 }];
  const fit = fitPca(train);
  const ext: InputPoint[] = [{ id: "e1", x: 10, y: -3 }, { id: "e2", x: -5, y: 8 }];
  const rows = projectPoints(fit, ext, 2);
  for (const [i, r] of rows.entries()) {
    const dx = ext[i].x - fit.mean.x, dy = ext[i].y - fit.mean.y;
    const s1 = dx * fit.axis1.x + dy * fit.axis1.y;
    const s2 = dx * fit.axis2.x + dy * fit.axis2.y;
    assert.ok(closeTo(r.score1, s1) && closeTo(r.score2, s2));
    assert.ok(closeTo(r.recX, fit.mean.x + s1 * fit.axis1.x + s2 * fit.axis2.x));
    assert.ok(closeTo(r.recY, fit.mean.y + s1 * fit.axis1.y + s2 * fit.axis2.y));
  }
});

test("符号规则：第一个显著非零分量为正", () => {
  const pts = [{ id: "1", x: 0, y: -5 }, { id: "2", x: 0, y: 0 }, { id: "3", x: 0, y: 5 }];
  const fit = fitPca(pts);
  assert.ok(fit.axis1.y > 0, "y 方向主轴取正");
});

test("非法输入拒绝", () => {
  assert.throws(() => fitPca([{ id: "a", x: 0, y: 0 }]), (e: unknown) => e instanceof PcaError && (e as PcaError).code === "INVALID_INPUT");
  assert.throws(() => fitPca([{ id: "a", x: NaN, y: 0 }, { id: "b", x: 1, y: 1 }]), /非有限/);
  assert.throws(() => fitPca([{ id: "a", x: Infinity, y: 0 }, { id: "b", x: 1, y: 1 }]), /非有限/);
  assert.throws(() => fitPca([{ id: "a", x: 1e13, y: 0 }, { id: "b", x: 1, y: 1 }]), /超出允许范围/);
  assert.throws(() => fitPca([{ id: "a", x: 0, y: 0 }, { id: "a", x: 1, y: 1 }]), /重复/);
  assert.throws(() => fitPca([{ id: "a", x: 0 } as never, { id: "b", x: 1, y: 1 }]), /缺失或非有限/);
});

test("斜线 45° 主轴", () => {
  const pts = [{ id: "1", x: 0, y: 0 }, { id: "2", x: 1, y: 1 }, { id: "3", x: 2, y: 2 }, { id: "4", x: 3, y: 3 }];
  const fit = fitPca(pts);
  const s = Math.SQRT1_2;
  assert.ok(closeTo(fit.axis1.x, s, 1e-9) && closeTo(fit.axis1.y, s, 1e-9));
  assert.ok(closeTo(fit.lambda2, 0, 1e-12));
});
