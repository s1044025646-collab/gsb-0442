/** CLI：add-sample / samples / fit / models / axes / project / verify / export / demo / serve */
import * as fs from "node:fs";
import { Store, defaultDbPath } from "./store";
import { fitPca, projectPoints, verifyModel, InputPoint, PcaError } from "./pca";
import { startServer } from "./server";
import { runDemo } from "./demo";

function fmt(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v !== 0 && (Math.abs(v) >= 1e6 || Math.abs(v) < 1e-6)) return v.toExponential(6);
  return String(Number(v.toPrecision(10)));
}

function readPointsFile(file: string): InputPoint[] {
  const raw = JSON.parse(fs.readFileSync(file, "utf8"));
  return (Array.isArray(raw) ? raw : raw.points) as InputPoint[];
}

function printModel(m: ReturnType<Store["getModel"]>): void {
  console.log(`模型 #${m.id}  (样本 #${m.sample_id}, 拟合于 ${m.created_at})`);
  console.log(`  均值 mean        = (${fmt(m.mean.x)}, ${fmt(m.mean.y)})`);
  console.log(`  协方差 cov       = [[${fmt(m.cov.a)}, ${fmt(m.cov.b)}], [${fmt(m.cov.b)}, ${fmt(m.cov.c)}]]  (分母 n-1)`);
  console.log(`  特征值 lambda    = (${fmt(m.lambda1)}, ${fmt(m.lambda2)})`);
  console.log(`  主轴1 axis1      = (${fmt(m.axis1.x)}, ${fmt(m.axis1.y)})`);
  console.log(`  主轴2 axis2      = (${fmt(m.axis2.x)}, ${fmt(m.axis2.y)})`);
  console.log(`  解释方差比例     = [${fmt(m.ratio1)}, ${fmt(m.ratio2)}]  (分母为 λ1+λ2)`);
  console.log(`  退化情形         = ${m.degenerate}, 容差 tol=${fmt(m.tol)}`);
  for (const note of m.notes) console.log(`  说明: ${note}`);
}

function printRows(rows: ReturnType<typeof projectPoints>, k: number): void {
  console.log("  id                | score1             | score2             | recX               | recY               | err^2");
  console.log("  " + "-".repeat(118));
  for (const r of rows) {
    const s1 = fmt(r.score1).padEnd(18), s2 = (k === 2 ? fmt(r.score2) : "-").padEnd(18);
    console.log(`  ${r.id.padEnd(18)}| ${s1}| ${s2}| ${fmt(r.recX).padEnd(18)}| ${fmt(r.recY).padEnd(18)}| ${fmt(r.err2)}`);
  }
  console.log(`  总平方误差 SSE = ${fmt(rows.reduce((s, r) => s + r.err2, 0))}`);
}

function printVerify(report: ReturnType<typeof verifyModel>): void {
  console.log(`核验结果: ${report.ok ? "全部通过" : "存在失败项"}`);
  for (const c of report.checks) console.log(`  [${c.ok ? "OK" : "FAIL"}] ${c.name}: ${c.detail}`);
}

const USAGE = `用法: pca <命令> [参数]
  serve [--port N]                 启动 HTTP API（默认自动选择空闲端口）
  add-sample <名称> <points.json>  保存样本新版本（文件为 [{id,x,y},...] 或 {points:[...]}）
  samples                          列出样本版本
  fit <sampleId>                   拟合二维 PCA 模型并保存
  models                           列出模型
  axes <modelId>                   查看均值/协方差/主轴/特征值
  project <modelId> --k 1|2 [--file points.json | --train] [--save]
  verify <modelId>                 独立核验分解与重建误差
  export <modelId> [out.json]      导出完整报告 JSON
  demo                             运行内置演示（已知例、斜线、各向同性、旧模型 vs 重新拟合）
数据库路径: 环境变量 PCA_DB，默认 ./data/pca.db`;

export async function main(argv: string[]): Promise<number> {
  const [cmd, ...args] = argv;
  if (!cmd || cmd === "help" || cmd === "--help") { console.log(USAGE); return cmd ? 0 : 1; }
  const store = new Store(defaultDbPath());
  try {
    switch (cmd) {
      case "serve": {
        const pi = args.indexOf("--port");
        const port = pi >= 0 ? Number(args[pi + 1]) : Number(process.env.PCA_PORT ?? 0);
        const { port: actual } = await startServer(store, port);
        console.log(`API 已启动: http://127.0.0.1:${actual}  (Ctrl+C 退出)`);
        return await new Promise<number>(() => { /* 保持运行 */ });
      }
      case "add-sample": {
        const [name, file] = args;
        if (!name || !file) throw new PcaError("INVALID_INPUT", "用法: add-sample <名称> <points.json>");
        const points = readPointsFile(file);
        const id = store.saveSample(name, points);
        console.log(`已保存样本 #${id}（${name}，${points.length} 点）`);
        return 0;
      }
      case "samples": {
        for (const s of store.listSamples()) console.log(`  #${s.id}  ${s.name}  n=${s.n}  ${s.created_at}`);
        return 0;
      }
      case "fit": {
        const sampleId = Number(args[0]);
        if (!Number.isInteger(sampleId)) throw new PcaError("INVALID_INPUT", "用法: fit <sampleId>");
        const points = store.getSamplePoints(sampleId);
        const fit = fitPca(points);
        const id = store.saveModel(sampleId, fit);
        printModel(store.getModel(id));
        return 0;
      }
      case "models": {
        for (const m of store.listModels()) console.log(`  #${m.id}  样本#${m.sample_id}  λ=(${fmt(m.lambda1)}, ${fmt(m.lambda2)})  ${m.degenerate}  ${m.created_at}`);
        return 0;
      }
      case "axes": {
        printModel(store.getModel(Number(args[0])));
        return 0;
      }
      case "project": {
        const modelId = Number(args[0]);
        const ki = args.indexOf("--k");
        const k = ki >= 0 ? Number(args[ki + 1]) : 2;
        if (k !== 1 && k !== 2) throw new PcaError("INVALID_INPUT", "--k 必须为 1 或 2");
        const m = store.getModel(modelId);
        const fi = args.indexOf("--file");
        const useTrain = args.includes("--train") || fi < 0;
        const points = useTrain ? store.getSamplePoints(m.sample_id) : readPointsFile(args[fi + 1]);
        const rows = projectPoints(m, points, k as 1 | 2);
        if (args.includes("--save")) store.saveProjections(modelId, k, rows);
        console.log(`模型 #${modelId} 投影 ${points.length} 点，保留 k=${k} 维，解释方差比例=${fmt(k === 1 ? m.ratio1 : m.ratio1 + m.ratio2)}`);
        printRows(rows, k);
        return 0;
      }
      case "verify": {
        const m = store.getModel(Number(args[0]));
        printVerify(verifyModel(m, store.getSamplePoints(m.sample_id)));
        return 0;
      }
      case "export": {
        const m = store.getModel(Number(args[0]));
        const out = {
          model: { id: m.id, sampleId: m.sample_id, mean: m.mean, cov: m.cov, lambda1: m.lambda1, lambda2: m.lambda2, axis1: m.axis1, axis2: m.axis2, tol: m.tol, explainedRatio: [m.ratio1, m.ratio2], degenerate: m.degenerate, notes: m.notes },
          trainPoints: store.getSamplePoints(m.sample_id),
          projections: store.listProjections(m.id),
          verify: verifyModel(m, store.getSamplePoints(m.sample_id)),
        };
        const text = JSON.stringify(out, null, 2);
        if (args[1]) { fs.writeFileSync(args[1], text); console.log(`已导出到 ${args[1]}`); }
        else console.log(text);
        return 0;
      }
      case "demo": {
        runDemo(store);
        return 0;
      }
      default:
        console.error(`未知命令: ${cmd}\n${USAGE}`);
        return 1;
    }
  } catch (err) {
    if (err instanceof PcaError) { console.error(`错误 [${err.code}] ${err.message}`); return 2; }
    throw err;
  } finally {
    if (cmd !== "serve") store.close();
  }
}

if (require.main === module) {
  main(process.argv.slice(2)).then((code) => { if (code !== 0) process.exitCode = code; })
    .catch((err) => { console.error(err); process.exitCode = 1; });
}
