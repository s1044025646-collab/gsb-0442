// 二维 PCA 核心数值计算：中心化、2x2 对称协方差闭式特征分解、投影与重建。
// 不调用任何现成 PCA 实现。

export interface Point {
  id: string;
  x: number;
  y: number;
}

export interface Vec2 {
  x: number;
  y: number;
}

export interface PcaModel {
  n: number;
  mean: Vec2;
  // 对称协方差矩阵 [[c11, c12], [c12, c22]]，分母 n-1
  cov: { c11: number; c12: number; c22: number };
  eigenvalues: [number, number]; // 降序
  axes: [Vec2, Vec2]; // 与 eigenvalues 对应的单位特征向量
  notes: string[];
  tolerance: number;
}

export class PcaError extends Error {
  constructor(
    public code: string,
    message: string,
  ) {
    super(message);
    this.name = "PcaError";
  }
}

export const MAX_POINTS = 100_000;
export const MAX_ABS_COORD = 1e12;
// 数值容差（公开固定值）：用于特征值归零、退化判定与核验残差
export const DEFAULT_TOLERANCE = 1e-10;
const SIGN_EPS = 1e-12;

export function validatePoints(points: unknown, minCount: number = 2): Point[] {
  if (!Array.isArray(points)) {
    throw new PcaError("INVALID_INPUT", "points 必须是数组");
  }
  if (points.length < minCount) {
    throw new PcaError("TOO_FEW_POINTS", `至少需要 ${minCount} 个点`);
  }
  if (points.length > MAX_POINTS) {
    throw new PcaError("TOO_MANY_POINTS", `样本数超过上限 ${MAX_POINTS}`);
  }
  const seen = new Set<string>();
  return points.map((raw, i) => {
    const p = raw as Partial<Point>;
    const id = p.id === undefined || p.id === null ? `p${i}` : String(p.id);
    if (seen.has(id)) {
      throw new PcaError("DUPLICATE_ID", `点编号重复: ${id}`);
    }
    seen.add(id);
    for (const key of ["x", "y"] as const) {
      const v = p[key];
      if (typeof v !== "number" || !Number.isFinite(v)) {
        throw new PcaError("NON_FINITE_VALUE", `点 ${id} 的 ${key} 缺失或非有限数值`);
      }
      if (Math.abs(v) > MAX_ABS_COORD) {
        throw new PcaError("COORD_OUT_OF_RANGE", `点 ${id} 的 ${key} 超过绝对值上限 ${MAX_ABS_COORD}`);
      }
    }
    return { id, x: p.x as number, y: p.y as number };
  });
}

export function computeMean(points: Point[]): Vec2 {
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
  }
  return { x: sx / points.length, y: sy / points.length };
}

// 手算协方差：先中心化，再外积累加，分母 n-1。
export function computeCovariance(points: Point[], mean: Vec2): { c11: number; c12: number; c22: number } {
  let c11 = 0;
  let c12 = 0;
  let c22 = 0;
  for (const p of points) {
    const dx = p.x - mean.x;
    const dy = p.y - mean.y;
    c11 += dx * dx;
    c12 += dx * dy;
    c22 += dy * dy;
  }
  const d = points.length - 1;
  return { c11: c11 / d, c12: c12 / d, c22: c22 / d };
}

function normalizeSign(v: Vec2): Vec2 {
  // 固定符号规则：第一个绝对值显著非零的分量取正
  if (Math.abs(v.x) > SIGN_EPS) return v.x > 0 ? v : { x: -v.x, y: -v.y };
  if (Math.abs(v.y) > SIGN_EPS) return v.y > 0 ? v : { x: -v.x, y: -v.y };
  return v;
}

function normalize(v: Vec2): Vec2 {
  const len = Math.hypot(v.x, v.y);
  if (len === 0) throw new PcaError("EIG_DECOMPOSITION_FAILED", "特征向量长度为零");
  return { x: v.x / len, y: v.y / len };
}

// 2x2 对称矩阵闭式特征分解，返回降序特征值与单位特征向量。
export function eigenDecompose(
  cov: { c11: number; c12: number; c22: number },
  tolerance: number = DEFAULT_TOLERANCE,
): { eigenvalues: [number, number]; axes: [Vec2, Vec2]; notes: string[] } {
  const { c11: a, c12: b, c22: c } = cov;
  const notes: string[] = [];
  const scale = Math.max(Math.abs(a), Math.abs(b), Math.abs(c), 1e-300);

  const tr = (a + c) / 2;
  const disc = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  let l1 = tr + disc;
  let l2 = tr - disc;

  // 数值误差产生的极小负特征值按公开容差归零；明显负值报错
  for (const [i, l] of [l1, l2].entries()) {
    if (l < 0) {
      if (l < -tolerance * scale) {
        throw new PcaError(
          "NEGATIVE_EIGENVALUE",
          `特征值 λ${i + 1}=${l} 明显为负，协方差矩阵异常`,
        );
      }
      if (i === 0) l1 = 0;
      else l2 = 0;
    }
  }

  let v1: Vec2;
  let v2: Vec2;
  const totalVar = l1 + l2;
  if (totalVar <= tolerance * scale && Math.abs(b) <= tolerance * scale && disc <= tolerance * scale) {
    // 全零方差：采用公开固定坐标轴
    v1 = { x: 1, y: 0 };
    v2 = { x: 0, y: 1 };
    notes.push("zero_total_variance: 总方差为零，使用固定坐标轴 e1=(1,0), e2=(0,1)");
  } else if (disc <= tolerance * scale * Math.max(1, Math.abs(tr))) {
    // 相等特征值（各向同性）：主轴不唯一，使用确定基
    v1 = { x: 1, y: 0 };
    v2 = { x: 0, y: 1 };
    notes.push("equal_eigenvalues: 特征值相等，主轴不唯一，使用确定基 e1=(1,0), e2=(0,1)");
  } else {
    // λ 的特征向量：取 (b, λ-a) 与 (λ-c, b) 中范数较大者，保证数值稳定
    const cand1 = { x: b, y: l1 - a };
    const cand2 = { x: l1 - c, y: b };
    v1 = normalize(
      Math.hypot(cand1.x, cand1.y) >= Math.hypot(cand2.x, cand2.y) ? cand1 : cand2,
    );
    // 与 v1 正交
    v2 = { x: -v1.y, y: v1.x };
  }

  v1 = normalizeSign(v1);
  v2 = normalizeSign(v2);
  return { eigenvalues: [l1, l2], axes: [v1, v2], notes };
}

export function fitPca(points: Point[], tolerance: number = DEFAULT_TOLERANCE): PcaModel {
  const mean = computeMean(points);
  const cov = computeCovariance(points, mean);
  const { eigenvalues, axes, notes } = eigenDecompose(cov, tolerance);
  return { n: points.length, mean, cov, eigenvalues, axes, notes, tolerance };
}

// 得分 = (点 - 训练均值) 与各主轴的内积；绝不用新点重新拟合均值。
export function projectPoint(model: PcaModel, p: Vec2, k: 1 | 2): number[] {
  const dx = p.x - model.mean.x;
  const dy = p.y - model.mean.y;
  const scores: number[] = [];
  for (let i = 0; i < k; i++) {
    scores.push(dx * model.axes[i].x + dy * model.axes[i].y);
  }
  return scores;
}

// 重建 = 保留得分 × 主轴 + 原均值
export function reconstructPoint(model: PcaModel, scores: number[]): Vec2 {
  let x = model.mean.x;
  let y = model.mean.y;
  for (let i = 0; i < scores.length; i++) {
    x += scores[i] * model.axes[i].x;
    y += scores[i] * model.axes[i].y;
  }
  return { x, y };
}

export function explainedVarianceRatios(model: PcaModel): { ratios: (number | null)[]; note: string | null } {
  const [l1, l2] = model.eigenvalues;
  const total = l1 + l2;
  if (total <= 0) {
    return { ratios: [null, null], note: "zero_total_variance: 总方差为零，解释方差比例无定义" };
  }
  return { ratios: [l1 / total, l2 / total], note: null };
}

export interface VerificationResult {
  ok: boolean;
  checks: { name: string; maxResidual: number; ok: boolean }[];
}

// 独立核验：正交归一、C·v = λ·v、训练点 k=2 重建误差
export function verifyModel(model: PcaModel, points: Point[]): VerificationResult {
  const scale = Math.max(
    Math.abs(model.cov.c11),
    Math.abs(model.cov.c12),
    Math.abs(model.cov.c22),
    1,
  );
  const checks: VerificationResult["checks"] = [];

  // 1. 主轴单位长度
  for (let i = 0; i < 2; i++) {
    const len = Math.hypot(model.axes[i].x, model.axes[i].y);
    const r = Math.abs(len - 1);
    checks.push({ name: `axis${i + 1}_unit_norm`, maxResidual: r, ok: r <= 1e-9 });
  }
  // 2. 主轴正交
  {
    const dot = model.axes[0].x * model.axes[1].x + model.axes[0].y * model.axes[1].y;
    checks.push({ name: "axes_orthogonal", maxResidual: Math.abs(dot), ok: Math.abs(dot) <= 1e-9 });
  }
  // 3. C·v = λ·v
  for (let i = 0; i < 2; i++) {
    const v = model.axes[i];
    const l = model.eigenvalues[i];
    const rx = model.cov.c11 * v.x + model.cov.c12 * v.y - l * v.x;
    const ry = model.cov.c12 * v.x + model.cov.c22 * v.y - l * v.y;
    const r = Math.hypot(rx, ry);
    checks.push({
      name: `eigen_equation_axis${i + 1}`,
      maxResidual: r,
      ok: r <= 1e-8 * scale,
    });
  }
  // 4. 保留两维时训练点重建误差
  {
    let maxErr = 0;
    for (const p of points) {
      const scores = projectPoint(model, p, 2);
      const rec = reconstructPoint(model, scores);
      maxErr = Math.max(maxErr, Math.hypot(rec.x - p.x, rec.y - p.y));
    }
    const coordScale = Math.max(1, ...points.map((p) => Math.hypot(p.x, p.y)));
    checks.push({
      name: "full_reconstruction",
      maxResidual: maxErr,
      ok: maxErr <= 1e-8 * coordScale,
    });
  }
  return { ok: checks.every((c) => c.ok), checks };
}

