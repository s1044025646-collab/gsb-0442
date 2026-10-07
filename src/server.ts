/** HTTP API：样本版本、拟合、查询主轴、投影、重建、核验、导出。 */
import * as http from "node:http";
import { Store } from "./store";
import { fitPca, projectPoints, verifyModel, validatePoints, PcaError, InputPoint } from "./pca";

const STATUS_BY_CODE: Record<string, number> = {
  INVALID_INPUT: 400, NOT_FOUND: 404, NUMERIC_FAILURE: 422, DB_ERROR: 500,
};

function send(res: http.ServerResponse, status: number, body: unknown): void {
  const s = JSON.stringify(body, null, 2);
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(s);
}

async function readBody(req: http.IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += (chunk as Buffer).length;
    if (size > 10 * 1024 * 1024) throw new PcaError("INVALID_INPUT", "请求体超过 10MB 上限");
    chunks.push(chunk as Buffer);
  }
  if (chunks.length === 0) return {};
  try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); }
  catch { throw new PcaError("INVALID_INPUT", "请求体不是合法 JSON"); }
}

export function modelReport(v: ReturnType<Store["getModel"]>) {
  return {
    id: v.id, sampleId: v.sample_id, mean: v.mean, cov: v.cov,
    lambda1: v.lambda1, lambda2: v.lambda2, axis1: v.axis1, axis2: v.axis2,
    tol: v.tol, totalVariance: v.total_variance,
    explainedRatio: [v.ratio1, v.ratio2], degenerate: v.degenerate,
    notes: v.notes, createdAt: v.created_at,
  };
}

export function createServer(store: Store): http.Server {
  return http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://localhost");
      const p = url.pathname;
      const method = req.method ?? "GET";

      if (method === "GET" && p === "/health") return send(res, 200, { ok: true });
      if (method === "GET" && p === "/samples") return send(res, 200, { samples: store.listSamples() });
      if (method === "GET" && p === "/models") return send(res, 200, { models: store.listModels() });

      if (method === "POST" && p === "/samples") {
        const body = await readBody(req) as Record<string, unknown>;
        const points = body.points as InputPoint[];
        validatePoints(points); // 先校验，失败不触碰数据库
        const id = store.saveSample(typeof body.name === "string" ? body.name : `sample-${Date.now()}`, points);
        return send(res, 201, { sampleId: id, n: points.length });
      }

      if (method === "POST" && p === "/models") {
        const body = await readBody(req) as Record<string, unknown>;
        const sampleId = Number(body.sampleId);
        if (!Number.isInteger(sampleId)) throw new PcaError("INVALID_INPUT", "sampleId 必须为整数");
        const points = store.getSamplePoints(sampleId);
        const fit = fitPca(points);
        const id = store.saveModel(sampleId, fit);
        return send(res, 201, { model: modelReport(store.getModel(id)) });
      }

      const mModel = p.match(/^\/models\/(\d+)(?:\/(project|verify|export))?$/);
      if (mModel) {
        const modelId = Number(mModel[1]);
        const sub = mModel[2];
        const m = store.getModel(modelId);
        if (method === "GET" && !sub) return send(res, 200, { model: modelReport(m) });
        if (method === "GET" && sub === "verify") {
          const train = store.getSamplePoints(m.sample_id);
          return send(res, 200, { modelId, report: verifyModel(m, train) });
        }
        if (method === "GET" && sub === "export") {
          const train = store.getSamplePoints(m.sample_id);
          return send(res, 200, {
            model: modelReport(m), trainPoints: train,
            projections: store.listProjections(modelId),
            verify: verifyModel(m, train),
          });
        }
        if (method === "POST" && sub === "project") {
          const body = await readBody(req) as Record<string, unknown>;
          const k = body.k === undefined ? 2 : Number(body.k);
          if (k !== 1 && k !== 2) throw new PcaError("INVALID_INPUT", "k 必须为 1 或 2");
          const points = body.train === true
            ? store.getSamplePoints(m.sample_id)
            : (body.points as InputPoint[]);
          validatePoints(points, 1);
          const rows = projectPoints(m, points, k as 1 | 2);
          if (body.save === true) store.saveProjections(modelId, k, rows);
          const sse = rows.reduce((s, r) => s + r.err2, 0);
          return send(res, 200, {
            modelId, k,
            explainedVarianceRatio: k === 1 ? m.ratio1 : m.ratio1 + m.ratio2,
            totalSquaredError: sse, rows,
          });
        }
      }
      throw new PcaError("NOT_FOUND", `未知路由: ${method} ${p}`);
    } catch (err) {
      if (err instanceof PcaError) {
        return send(res, STATUS_BY_CODE[err.code] ?? 500, { error: { code: err.code, message: err.message } });
      }
      return send(res, 500, { error: { code: "INTERNAL", message: (err as Error).message } });
    }
  });
}

/** 默认端口 0 表示由系统分配空闲端口；可用 PCA_PORT 或参数指定。 */
export async function startServer(store: Store, port = Number(process.env.PCA_PORT ?? 0)): Promise<{ server: http.Server; port: number }> {
  const server = createServer(store);
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  const addr = server.address();
  const actual = typeof addr === "object" && addr ? addr.port : port;
  return { server, port: actual };
}



