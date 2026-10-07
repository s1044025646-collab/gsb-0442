// CLI：fit / axes / project / reconstruct / verify / report / list / demo
import { openDb, saveDataset, fitAndSaveModel, loadModel, loadDatasetPoints, saveProjection, listDatasets, listModels, DB_PATH } from "./db";
import { PcaError, Point, validatePoints, verifyModel, explainedVarianceRatios, reconstructPoint, PcaModel } from "./pca";
import { buildProjectionReport } from "./report";
import * as fs from "node:fs";

function fmt(v: number | null, digits = 6): string {
  if (v === null) return "n/a";
  if (!Number.isFinite(v)) return String(v);
  return Math.abs(v) < 1e-12 ? "0" : Number(v.toPrecision(digits)).toString();
}

function printModel(model: PcaModel): void {
  console.log(`样本数 n        : ${model.n}`);
  console.log(`均值 mean       : (${fmt(model.mean.x)}, ${fmt(model.mean.y)})`);
  console.log(`协方差矩阵 (n-1): [[${fmt(model.cov.c11)}, ${fmt(model.cov.c12)}], [${fmt(model.cov.c12)}, ${fmt(model.cov.c22)}]]`);
  console.log(`特征值 λ1, λ2   : ${fmt(model.eigenvalues[0])}, ${fmt(model.eigenvalues[1])}`);
  console.log(`主轴1 (单位)    : (${fmt(model.axes[0].x)}, ${fmt(model.axes[0].y)})`);
  console.log(`主轴2 (单位)    : (${fmt(model.axes[1].x)}, ${fmt(model.axes[1].y)})`);
  const { ratios, note } = explainedVarianceRatios(model);
  console.log(`解释方差比例    : [${fmt(ratios[0])}, ${fmt(ratios[1])}]${note ? `  (${note})` : ""}`);
  for (const n of model.notes) console.log(`说明            : ${n}`);
  console.log(`容差            : ${model.tolerance}`);
}

function printProjectionTable(rows: { id: string; x: number; y: number; scores: number[]; rec: { x: number; y: number }; sqError: number }[], k: number): void {
  const scoreCols = k === 2 ? "score1        score2        " : "score1        ";
  console.log(`id        x             y             ${scoreCols}rec_x         rec_y         sq_error`);
  for (const r of rows) {
    const s = r.scores.map((v) => fmt(v).padEnd(14)).join("");
    console.log(
      `${r.id.padEnd(10)}${fmt(r.x).padEnd(14)}${fmt(r.y).padEnd(14)}${s}${fmt(r.rec.x).padEnd(14)}${fmt(r.rec.y).padEnd(14)}${fmt(r.sqError)}`,
    );
  }
}

function loadPointsArg(arg: string): unknown {
  if (fs.existsSync(arg)) return JSON.parse(fs.readFileSync(arg, "utf8"));
  return JSON.parse(arg);
}

function usage(): void {
  console.log(`用法: node dist/src/cli.js <命令> [参数]
  fit <name> <pointsJSON|file>            保存样本版本并拟合模型
  axes <modelId>                          查询主轴
  project <modelId> [pointsJSON|file] [k] 投影（缺省为训练点，k 缺省 2）
  reconstruct <modelId> <scoresJSON>      由得分重建，如 "[[1,2],[3]]"
  verify <modelId>                        独立核验分解与重建
  report <modelId>                        导出完整报告 JSON
  list                                    列出数据集与模型
  demo                                    运行内置演示`);
}

function main(): void {
  const [cmd, ...args] = process.argv.slice(2);
  if (!cmd) return usage();
  const db = openDb();

  switch (cmd) {
    case "fit": {
      const [name, pointsArg] = args;
      if (!name || !pointsArg) throw new PcaError("INVALID_INPUT", "用法: fit <name> <pointsJSON|file>");
      const { datasetId } = saveDataset(db, name, loadPointsArg(pointsArg));
      const { modelId, model } = fitAndSaveModel(db, datasetId);
      console.log(`数据集 #${datasetId} 已保存，模型 #${modelId} 拟合完成`);
      printModel(model);
      break;
    }
    case "axes": {
      const { model } = loadModel(db, Number(args[0]));
      printModel(model);
      break;
    }
    case "project": {
      const modelId = Number(args[0]);
      const { model, datasetId } = loadModel(db, modelId);
      const points = args[1] && !/^[12]$/.test(args[1])
        ? validatePoints(loadPointsArg(args[1]), 1)
        : loadDatasetPoints(db, datasetId);
      const kArg = args[1] && /^[12]$/.test(args[1]) ? args[1] : args[2];
      const k = (kArg === "1" ? 1 : 2) as 1 | 2;
      const report = buildProjectionReport(model, points, k);
      saveProjection(db, modelId, k, report.rows);
      console.log(`模型 #${modelId}，保留 k=${k} 个主成分`);
      printProjectionTable(report.rows, k);
      console.log(`总平方误差: ${fmt(report.totalSqError)}`);
      console.log(`解释方差比例: [${fmt(report.ratios[0])}, ${fmt(report.ratios[1])}]${report.ratioNote ? ` (${report.ratioNote})` : ""}`);
      break;
    }
    case "reconstruct": {
      const { model } = loadModel(db, Number(args[0]));
      const scoresList = JSON.parse(args[1]) as number[][];
      for (const s of scoresList) {
        const rec = reconstructPoint(model, s);
        console.log(`scores [${s.map(fmt).join(", ")}] -> (${fmt(rec.x)}, ${fmt(rec.y)})`);
      }
      break;
    }
    case "verify": {
      const { model, datasetId } = loadModel(db, Number(args[0]));
      const points = loadDatasetPoints(db, datasetId);
      const v = verifyModel(model, points);
      for (const c of v.checks) {
        console.log(`${c.ok ? "PASS" : "FAIL"}  ${c.name}  maxResidual=${c.maxResidual.toExponential(3)}`);
      }
      console.log(v.ok ? "核验通过" : "核验失败");
      if (!v.ok) process.exitCode = 1;
      break;
    }
    case "report": {
      const { model, datasetId } = loadModel(db, Number(args[0]));
      const points = loadDatasetPoints(db, datasetId);
      console.log(JSON.stringify({
        modelId: Number(args[0]), datasetId, model,
        explainedVarianceRatios: explainedVarianceRatios(model),
        trainingPoints: points,
        verification: verifyModel(model, points),
      }, null, 2));
      break;
    }
    case "list": {
      console.log("数据集:", JSON.stringify(listDatasets(db)));
      console.log("模型  :", JSON.stringify(listModels(db)));
      break;
    }
    case "demo":
      demo();
      break;
    default:
      usage();
  }
}

function demo(): void {
  const db = openDb();
  console.log("=== 演示 1: 水平三点 (-1,0),(0,0),(1,0) ===");
  const line: Point[] = [
    { id: "a", x: -1, y: 0 },
    { id: "b", x: 0, y: 0 },
    { id: "c", x: 1, y: 0 },
  ];
  const d1 = saveDataset(db, "demo-line3", line);
  const m1 = fitAndSaveModel(db, d1.datasetId);
  printModel(m1.model);
  console.log("--- 保留 k=1 投影训练点（应无损重建） ---");
  const r1 = buildProjectionReport(m1.model, line, 1);
  printProjectionTable(r1.rows, 1);
  console.log(`总平方误差: ${fmt(r1.totalSqError)}`);

  console.log("\n=== 演示 2: 斜线点（主轴旋转 45°） ===");
  const diag: Point[] = [
    { id: "p1", x: 0, y: 0 },
    { id: "p2", x: 1, y: 1 },
    { id: "p3", x: 2, y: 2 },
    { id: "p4", x: 3, y: 3 },
  ];
  const d2 = saveDataset(db, "demo-diag", diag);
  const m2 = fitAndSaveModel(db, d2.datasetId);
  printModel(m2.model);

  console.log("\n=== 演示 3: 各向同性四点（重复特征值，主轴不唯一） ===");
  const iso: Point[] = [
    { id: "q1", x: 1, y: 0 },
    { id: "q2", x: -1, y: 0 },
    { id: "q3", x: 0, y: 1 },
    { id: "q4", x: 0, y: -1 },
  ];
  const d3 = saveDataset(db, "demo-iso4", iso);
  const m3 = fitAndSaveModel(db, d3.datasetId);
  printModel(m3.model);

  console.log("\n=== 演示 4: 旧模型投影新点 vs 重新拟合 ===");
  const train: Point[] = [
    { id: "t1", x: 0, y: 0 },
    { id: "t2", x: 2, y: 1 },
    { id: "t3", x: 4, y: 2 },
    { id: "t4", x: 6, y: 3 },
  ];
  const dOld = saveDataset(db, "demo-old", train);
  const mOld = fitAndSaveModel(db, dOld.datasetId);
  const newPoints: Point[] = [
    { id: "n1", x: 8, y: 4 },
    { id: "n2", x: 10, y: 6 },
  ];
  console.log(`旧模型 #${mOld.modelId}（均值 (${fmt(mOld.model.mean.x)}, ${fmt(mOld.model.mean.y)})）投影新点:`);
  const rOld = buildProjectionReport(mOld.model, newPoints, 1);
  printProjectionTable(rOld.rows, 1);
  const all = [...train, ...newPoints];
  const dNew = saveDataset(db, "demo-refit", all);
  const mNew = fitAndSaveModel(db, dNew.datasetId);
  console.log(`重新拟合模型 #${mNew.modelId}（均值 (${fmt(mNew.model.mean.x)}, ${fmt(mNew.model.mean.y)})）投影同样新点:`);
  const rNew = buildProjectionReport(mNew.model, newPoints, 1);
  printProjectionTable(rNew.rows, 1);
  console.log("注意两者均值与主轴不同：投影始终使用训练均值，不用新点重新拟合。");

  console.log("\n=== 演示 5: 核验 ===");
  const v = verifyModel(m1.model, line);
  for (const c of v.checks) console.log(`${c.ok ? "PASS" : "FAIL"}  ${c.name}  maxResidual=${c.maxResidual.toExponential(3)}`);
  console.log(`\n数据库文件: ${DB_PATH}（重启后 report/axes 可复现历史结果）`);
}

try {
  main();
} catch (e) {
  if (e instanceof PcaError) {
    console.error(`错误 [${e.code}]: ${e.message}`);
    process.exit(2);
  }
  throw e;
}

