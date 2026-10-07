/** 二维 PCA 核心：中心化、样本协方差（n-1）、2x2 对称矩阵特征分解、投影与重建。 */
export interface InputPoint { id: string; x: number; y: number }
export interface Vec2 { x: number; y: number }

export const MAX_POINTS = 100_000;
export const MAX_ABS_COORD = 1e12;
/** 相对容差基准，用于特征值归零、重复特征值判定与核验。 */
export const REL_TOL = 1e-12;
/** 明显负特征值判定阈值（相对），超过则视为算法错误。 */
export const NEG_EIG_REL_TOL = 1e-9;

export class PcaError extends Error {
  constructor(public readonly code: string, message: string) {
    super(message);
    this.name = "PcaError";
  }
}

export interface FitResult {
  n: number;
  mean: Vec2;
  cov: { a: number; b: number; c: number }; // [[a,b],[b,c]]
  lambda1: number;
  lambda2: number;
  axis1: Vec2;
  axis2: Vec2;
  tol: number;
  totalVariance: number;
  explainedRatio: [number, number];
  degenerate: "none" | "zero-variance" | "repeated-eigenvalue";
  notes: string[];
}

export interface ProjectionRow {
  id: string;
  x: number;
  y: number;
  score1: number;
  score2: number;
  recX: number;
  recY: number;
  err2: number;
}

export interface ModelLike {
  mean: Vec2;
  cov: { a: number; b: number; c: number };
  lambda1: number;
  lambda2: number;
  axis1: Vec2;
  axis2: Vec2;
  tol: number;
}

export function validatePoints(points: unknown, minCount = 2): asserts points is InputPoint[] {
  if (!Array.isArray(points)) throw new PcaError("INVALID_INPUT", "points 必须是数组");
  if (points.length < minCount) throw new PcaError("INVALID_INPUT", `至少需要 ${minCount} 个点`);
  if (points.length > MAX_POINTS) throw new PcaError("INVALID_INPUT", `样本数超过上限 ${MAX_POINTS}`);
  const seen = new Set<string>();
  for (const p of points) {
    if (p === null || typeof p !== "object") throw new PcaError("INVALID_INPUT", "点必须是对象 {id,x,y}");
    const q = p as Record<string, unknown>;
    if (typeof q.id !== "string" || q.id.length === 0) throw new PcaError("INVALID_INPUT", "每个点需要非空字符串 id");
    if (seen.has(q.id)) throw new PcaError("INVALID_INPUT", `点 id 重复: ${q.id}`);
    seen.add(q.id);
    for (const key of ["x", "y"] as const) {
      const v = q[key];
      if (typeof v !== "number" || !Number.isFinite(v)) throw new PcaError("INVALID_INPUT", `点 ${q.id} 的 ${key} 缺失或非有限数值`);
      if (Math.abs(v) > MAX_ABS_COORD) throw new PcaError("INVALID_INPUT", `点 ${q.id} 的 ${key} 超出允许范围 ±${MAX_ABS_COORD}`);
    }
  }
}

export function mean2(points: InputPoint[]): Vec2 {
  let sx = 0, sy = 0;
  for (const p of points) { sx += p.x; sy += p.y; }
  return { x: sx / points.length, y: sy / points.length };
}

/** 样本协方差，分母 n-1，基于中心化外积直接累加。 */
export function covariance2(points: InputPoint[], mean: Vec2): { a: number; b: number; c: number } {
  let a = 0, b = 0, c = 0;
  for (const p of points) {
    const dx = p.x - mean.x, dy = p.y - mean.y;
    a += dx * dx; b += dx * dy; c += dy * dy;
  }
  const d = points.length - 1;
  return { a: a / d, b: b / d, c: c / d };
}

/** 符号规范：第一个绝对值显著非零的分量取正。 */
function signNormalize(v: Vec2): Vec2 {
  const z = (u: number) => (u === 0 ? 0 : u);
  v = { x: z(v.x), y: z(v.y) };
  const eps = 1e-300;
  if (Math.abs(v.x) > eps) return v.x < 0 ? { x: -v.x, y: -v.y } : v;
  if (Math.abs(v.y) > eps) return v.y < 0 ? { x: -v.x, y: -v.y } : v;
  return v;
}

function normalize(v: Vec2): Vec2 {
  const n = Math.hypot(v.x, v.y);
  return { x: v.x / n, y: v.y / n };
}

export interface EigenResult {
  lambda1: number; lambda2: number;
  axis1: Vec2; axis2: Vec2;
  repeated: boolean;
  scale: number;
}

/** 2x2 对称矩阵 [[a,b],[b,c]] 的解析特征分解，特征值降序。 */
export function eigen2x2(a: number, b: number, c: number): EigenResult {
  const scale = Math.max(1, Math.abs(a), Math.abs(b), Math.abs(c));
  const half = (a - c) / 2;
  const disc = Math.hypot(half, b);
  const tr = (a + c) / 2;
  let l1 = tr + disc;
  let l2 = tr - disc;
  // 数值误差导致的极小负特征值按公开容差归零；明显负值报错。
  for (const [i, l] of [l1, l2].entries()) {
    if (l < 0) {
      if (l < -NEG_EIG_REL_TOL * scale) {
        throw new PcaError("NUMERIC_FAILURE", `特征值明显为负 (${l})，分解不可信`);
      }
      if (i === 0) l1 = 0; else l2 = 0;
    }
  }
  const repeated = disc <= REL_TOL * scale;
  let axis1: Vec2;
  if (repeated) {
    // 特征值重复：主轴不唯一，使用公开确定基。
    axis1 = { x: 1, y: 0 };
  } else {
    // 对 λ1 取 (b, λ1-a) 与 (λ1-c, b) 中范数较大者，更稳定。
    const c1 = { x: b, y: l1 - a };
    const c2 = { x: l1 - c, y: b };
    const v = Math.hypot(c1.x, c1.y) >= Math.hypot(c2.x, c2.y) ? c1 : c2;
    axis1 = normalize(v);
  }
  axis1 = signNormalize(axis1);
  // 第二轴取第一轴的垂直方向，保证严格正交归一，再做符号规范。
  let axis2 = signNormalize({ x: -axis1.y, y: axis1.x });
  return { lambda1: l1, lambda2: l2, axis1, axis2, repeated, scale };
}

export function fitPca(points: InputPoint[]): FitResult {
  validatePoints(points);
  const mean = mean2(points);
  const cov = covariance2(points, mean);
  const { a, b, c } = cov;
  const scale = Math.max(1, Math.abs(a), Math.abs(b), Math.abs(c));
  const tol = REL_TOL * scale;
  const eig = eigen2x2(a, b, c);
  const total = eig.lambda1 + eig.lambda2;
  const notes: string[] = [];
  let degenerate: FitResult["degenerate"] = "none";
  let explainedRatio: [number, number];
  if (total <= tol) {
    degenerate = "zero-variance";
    explainedRatio = [0, 0];
    notes.push("总方差为零（所有点重合）：采用公开固定坐标轴 e1=(1,0), e2=(0,1)，解释方差比例按无方差口径返回 0。");
  } else {
    explainedRatio = [eig.lambda1 / total, eig.lambda2 / total];
    if (eig.repeated) {
      degenerate = "repeated-eigenvalue";
      notes.push("两个特征值相等：主轴方向不唯一，报告使用确定基 e1=(1,0), e2=(0,1)。");
    }
  }
  notes.push("坐标保留输入单位，未按坐标标准差缩放；尺度差异会影响主轴方向。");
  return {
    n: points.length, mean, cov,
    lambda1: eig.lambda1, lambda2: eig.lambda2,
    axis1: eig.axis1, axis2: eig.axis2,
    tol, totalVariance: total, explainedRatio, degenerate, notes,
  };
}

export function dot(u: Vec2, v: Vec2): number { return u.x * v.x + u.y * v.y; }

/** 投影：得分 = (p - 训练均值)·轴；重建 = 均值 + Σ 保留得分×轴。 */
export function projectPoints(model: ModelLike, points: InputPoint[], k: 1 | 2): ProjectionRow[] {
  validatePoints(points, 1);
  if (k !== 1 && k !== 2) throw new PcaError("INVALID_INPUT", "k 必须为 1 或 2");
  return points.map((p) => {
    const dx = p.x - model.mean.x, dy = p.y - model.mean.y;
    const score1 = dx * model.axis1.x + dy * model.axis1.y;
    const score2 = dx * model.axis2.x + dy * model.axis2.y;
    let rx = model.mean.x + score1 * model.axis1.x;
    let ry = model.mean.y + score1 * model.axis1.y;
    if (k === 2) { rx += score2 * model.axis2.x; ry += score2 * model.axis2.y; }
    const ex = p.x - rx, ey = p.y - ry;
    return { id: p.id, x: p.x, y: p.y, score1, score2, recX: rx, recY: ry, err2: ex * ex + ey * ey };
  });
}

export interface VerifyReport {
  ok: boolean;
  checks: { name: string; ok: boolean; detail: string }[];
}

/** 独立核验：正交归一、C·v=λ·v、谱分解重建、训练点重建误差恒等式。 */
export function verifyModel(model: ModelLike, trainPoints?: InputPoint[]): VerifyReport {
  const checks: VerifyReport["checks"] = [];
  const scale = Math.max(1, Math.abs(model.cov.a), Math.abs(model.cov.b), Math.abs(model.cov.c));
  const tol = Math.max(model.tol, REL_TOL * scale) * 100; // 核验容差略宽于拟合容差
  const push = (name: string, ok: boolean, detail: string) => checks.push({ name, ok, detail });

  const n1 = Math.hypot(model.axis1.x, model.axis1.y);
  const n2 = Math.hypot(model.axis2.x, model.axis2.y);
  const ip = dot(model.axis1, model.axis2);
  push("orthonormal", Math.abs(n1 - 1) <= 1e-9 && Math.abs(n2 - 1) <= 1e-9 && Math.abs(ip) <= 1e-9,
    `|axis1|=${n1}, |axis2|=${n2}, axis1·axis2=${ip}`);

  const { a, b, c } = model.cov;
  const r1x = a * model.axis1.x + b * model.axis1.y - model.lambda1 * model.axis1.x;
  const r1y = b * model.axis1.x + c * model.axis1.y - model.lambda1 * model.axis1.y;
  const r2x = a * model.axis2.x + b * model.axis2.y - model.lambda2 * model.axis2.x;
  const r2y = b * model.axis2.x + c * model.axis2.y - model.lambda2 * model.axis2.y;
  const res = Math.max(Math.hypot(r1x, r1y), Math.hypot(r2x, r2y));
  push("eigen-residual", res <= tol, `max ||C·v - λ·v|| = ${res} (tol=${tol})`);

  // 谱分解重建：C ≈ λ1 v1 v1ᵀ + λ2 v2 v2ᵀ
  const m11 = model.lambda1 * model.axis1.x ** 2 + model.lambda2 * model.axis2.x ** 2;
  const m12 = model.lambda1 * model.axis1.x * model.axis1.y + model.lambda2 * model.axis2.x * model.axis2.y;
  const m22 = model.lambda1 * model.axis1.y ** 2 + model.lambda2 * model.axis2.y ** 2;
  const specRes = Math.max(Math.abs(m11 - a), Math.abs(m12 - b), Math.abs(m22 - c));
  push("spectral-reconstruction", specRes <= tol, `max |C - Σλvvᵀ| = ${specRes} (tol=${tol})`);

  push("eigenvalue-order", model.lambda1 >= model.lambda2 - tol && model.lambda2 >= -NEG_EIG_REL_TOL * scale,
    `λ1=${model.lambda1}, λ2=${model.lambda2}`);

  if (trainPoints && trainPoints.length >= 2) {
    const rows2 = projectPoints(model, trainPoints, 2);
    const sse2 = rows2.reduce((s, r) => s + r.err2, 0);
    const scale2 = scale * trainPoints.length;
    push("reconstruct-k2", sse2 <= Math.max(tol * scale2, 1e-18), `保留两维的总平方误差 SSE=${sse2}`);
    const rows1 = projectPoints(model, trainPoints, 1);
    const sse1 = rows1.reduce((s, r) => s + r.err2, 0);
    const expect = (trainPoints.length - 1) * model.lambda2;
    const diff = Math.abs(sse1 - expect);
    push("reconstruct-k1-identity", diff <= Math.max(1e-6 * Math.max(1, expect), 1e-18),
      `SSE(k=1)=${sse1}, (n-1)·λ2=${expect}, |差|=${diff}`);
  }
  return { ok: checks.every((c) => c.ok), checks };
}



