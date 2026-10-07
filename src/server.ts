// HTTP API 服务（node:http，无外部依赖）
import * as http from "node:http";
import { openDb, saveDataset, listDatasets, fitAndSaveModel, loadModel, listModels, loadDatasetPoints, saveProjection, listProjections, DB_PATH } from "./db";
import { PcaError, validatePoints, verifyModel, explainedVarianceRatios, reconstructPoint } from "./pca";
import { buildProjectionReport } from "./report";

function send(res: http.ServerResponse, status: number, body: unknown): void {
  const json = JSON.stringify(body, null, 2);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(json);
}

function sendError(res: http.ServerResponse, e: unknown): void {
  if (e instanceof PcaError) {
    send(res, 400, { error: { code: e.code, message: e.message } });
  } else {
    send(res, 500, { error: { code: "INTERNAL_ERROR", message: (e as Error).message } });
  }
}

async function readBody(req: http.IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  if (chunks.length === 0) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new PcaError("INVALID_JSON", "请求体不是合法 JSON");
  }
}

function parseK(v: unknown): 1 | 2 {
  const k = v === undefined ? 2 : Number(v);
  if (k !== 1 && k !== 2) throw new PcaError("INVALID_K", "k 必须为 1 或 2");
  return k;
}

export function createServer(dbPath: string = DB_PATH): http.Server {
  const db = openDb(dbPath);
  return http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://localhost");
      const parts = url.pathname.split("/").filter(Boolean);
      const method = req.method ?? "GET";

      // GET /health
      if (method === "GET" && url.pathname === "/health") {
        return send(res, 200, { ok: true });
      }
      // POST /datasets {name, points:[{id,x,y}]}
      if (method === "POST" && url.pathname === "/datasets") {
        const body = (await readBody(req)) as { name?: string; points?: unknown };
        const { datasetId, points } = saveDataset(db, body.name ?? "dataset", body.points);
        return send(res, 201, { datasetId, n: points.length });
      }
      // GET /datasets
      if (method === "GET" && url.pathname === "/datasets") {
        return send(res, 200, { datasets: listDatasets(db) });
      }
      // POST /models {datasetId, tolerance?}
      if (method === "POST" && url.pathname === "/models") {
        const body = (await readBody(req)) as { datasetId?: number; tolerance?: number };
        if (typeof body.datasetId !== "number") throw new PcaError("INVALID_INPUT", "缺少 datasetId");
        const { modelId, model } = fitAndSaveModel(db, body.datasetId, body.tolerance);
        return send(res, 201, { modelId, datasetId: body.datasetId, model });
      }
      // GET /models
      if (method === "GET" && url.pathname === "/models") {
        return send(res, 200, { models: listModels(db) });
      }
      if (parts[0] === "models" && parts[1]) {
        const modelId = Number(parts[1]);
        if (!Number.isInteger(modelId)) throw new PcaError("INVALID_INPUT", "模型 id 非法");
        // GET /models/:id 查询主轴
        if (method === "GET" && parts.length === 2) {
          const { model, datasetId } = loadModel(db, modelId);
          const { ratios, note } = explainedVarianceRatios(model);
          return send(res, 200, { modelId, datasetId, model, explainedVarianceRatios: ratios, ratioNote: note });
        }
        // POST /models/:id/project {points?, k?} 缺省投影训练点
        if (method === "POST" && parts[2] === "project") {
          const body = (await readBody(req)) as { points?: unknown; k?: number };
          const { model, datasetId } = loadModel(db, modelId);
          const k = parseK(body.k);
          const points = body.points === undefined
            ? loadDatasetPoints(db, datasetId)
            : validatePoints(body.points, 1);
          const report = buildProjectionReport(model, points, k);
          saveProjection(db, modelId, k, report.rows);
          return send(res, 200, { modelId, k, ...report });
        }
        // POST /models/:id/reconstruct {scores:[[s1,s2?],...]}
        if (method === "POST" && parts[2] === "reconstruct") {
          const body = (await readBody(req)) as { scores?: number[][] };
          const { model, datasetId } = loadModel(db, modelId);
          if (!Array.isArray(body.scores)) throw new PcaError("INVALID_INPUT", "scores 必须是数组");
          const rows = body.scores.map((s, i) => {
            if (!Array.isArray(s) || s.length < 1 || s.length > 2 || s.some((v) => typeof v !== "number" || !Number.isFinite(v))) {
              throw new PcaError("INVALID_INPUT", `scores[${i}] 必须是 1~2 个有限数值`);
            }
            return { scores: s, rec: reconstructPoint(model, s) };
          });
          return send(res, 200, { modelId, datasetId, mean: model.mean, rows });
        }
        // GET /models/:id/verify 独立核验分解与重建
        if (method === "GET" && parts[2] === "verify") {
          const { model, datasetId } = loadModel(db, modelId);
          const points = loadDatasetPoints(db, datasetId);
          return send(res, 200, { modelId, ...verifyModel(model, points) });
        }
        // GET /models/:id/report 导出完整报告（含历史投影）
        if (method === "GET" && parts[2] === "report") {
          const { model, datasetId } = loadModel(db, modelId);
          const points = loadDatasetPoints(db, datasetId);
          const { ratios, note } = explainedVarianceRatios(model);
          return send(res, 200, {
            modelId, datasetId, model,
            explainedVarianceRatios: ratios, ratioNote: note,
            trainingPoints: points,
            verification: verifyModel(model, points),
            projections: listProjections(db, modelId),
          });
        }
      }
      send(res, 404, { error: { code: "NOT_FOUND", message: `${method} ${url.pathname} 不存在` } });
    } catch (e) {
      sendError(res, e);
    }
  });
}

if (require.main === module) {
  const portArg = process.argv.indexOf("--port");
  const port = portArg >= 0 ? Number(process.argv[portArg + 1]) : Number(process.env.PORT ?? 0);
  const server = createServer();
  server.listen(Number.isFinite(port) ? port : 0, () => {
    const addr = server.address();
    const p = typeof addr === "object" && addr ? addr.port : port;
    console.log(`PCA2D API 已启动: http://127.0.0.1:${p}  (数据库: ${DB_PATH})`);
  });
}


