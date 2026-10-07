// 报告组装：投影 + 重建 + 逐点误差 + 解释方差比例
import {
  PcaModel,
  Point,
  explainedVarianceRatios,
  projectPoint,
  reconstructPoint,
} from "./pca";

export interface ProjectionRow {
  id: string;
  x: number;
  y: number;
  scores: number[];
  rec: { x: number; y: number };
  sqError: number;
}

export function buildProjectionReport(
  model: PcaModel,
  points: Point[],
  k: 1 | 2,
): { rows: ProjectionRow[]; totalSqError: number; ratios: (number | null)[]; ratioNote: string | null; notes: string[] } {
  const rows: ProjectionRow[] = points.map((p) => {
    const scores = projectPoint(model, p, k);
    const rec = reconstructPoint(model, scores);
    const sqError = (rec.x - p.x) ** 2 + (rec.y - p.y) ** 2;
    return { id: p.id, x: p.x, y: p.y, scores, rec, sqError };
  });
  const totalSqError = rows.reduce((s, r) => s + r.sqError, 0);
  const { ratios, note } = explainedVarianceRatios(model);
  return { rows, totalSqError, ratios, ratioNote: note, notes: model.notes };
}
