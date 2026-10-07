/** 内置演示：已知例、斜线、各向同性四点、旧模型投影 vs 重新拟合、历史报告复现。 */
import { Store } from "./store";
import { fitPca, projectPoints, verifyModel, InputPoint } from "./pca";

function line(title: string): void { console.log("\n=== " + title + " ==="); }

function showFit(store: Store, name: string, points: InputPoint[], tag: string): number {
  const sid = store.saveSample(`${tag}-${name}`, points);
  const fit = fitPca(points);
  const mid = store.saveModel(sid, fit);
  const m = store.getModel(mid);
  console.log(`[${tag}] 模型 #${mid}: mean=(${m.mean.x}, ${m.mean.y})`);
  console.log(`  cov=[[${m.cov.a}, ${m.cov.b}], [${m.cov.b}, ${m.cov.c}]]  λ=(${m.lambda1}, ${m.lambda2})`);
  console.log(`  axis1=(${m.axis1.x}, ${m.axis1.y})  axis2=(${m.axis2.x}, ${m.axis2.y})  比例=[${m.ratio1}, ${m.ratio2}]  ${m.degenerate}`);
  return mid;
}

export function runDemo(store: Store): void {
  line("1. 已知例 (-1,0),(0,0),(1,0)：均值为原点，协方差 diag(1,0)，保留一维无损");
  const mid1 = showFit(store, "axis", [
    { id: "p1", x: -1, y: 0 }, { id: "p2", x: 0, y: 0 }, { id: "p3", x: 1, y: 0 },
  ], "demo");
  const m1 = store.getModel(mid1);
  const rows1 = projectPoints(m1, store.getSamplePoints(m1.sample_id), 1);
  for (const r of rows1) console.log(`  ${r.id}: score1=${r.score1} 重建=(${r.recX}, ${r.recY}) err^2=${r.err2}`);

  line("2. 斜线点 (0,0),(1,1),(2,2),(3,3)：主轴旋转 45°");
  showFit(store, "diag", [
    { id: "a", x: 0, y: 0 }, { id: "b", x: 1, y: 1 }, { id: "c", x: 2, y: 2 }, { id: "d", x: 3, y: 3 },
  ], "demo");

  line("3. 各向同性四点 (±1,±1)：特征值重复，主轴不唯一，使用确定基");
  showFit(store, "iso", [
    { id: "q1", x: 1, y: 1 }, { id: "q2", x: 1, y: -1 }, { id: "q3", x: -1, y: 1 }, { id: "q4", x: -1, y: -1 },
  ], "demo");

  line("4. 旧模型投影新点 vs 用新点重新拟合（二者均值与主轴不同）");
  const oldPoints: InputPoint[] = [
    { id: "o1", x: 0, y: 0 }, { id: "o2", x: 1, y: 0.1 }, { id: "o3", x: 2, y: -0.1 }, { id: "o4", x: 3, y: 0.05 },
  ];
  const oldMid = showFit(store, "old", oldPoints, "demo-old");
  const oldModel = store.getModel(oldMid);
  const newPoints: InputPoint[] = [
    { id: "n1", x: 10, y: 5 }, { id: "n2", x: 11, y: 5.2 }, { id: "n3", x: 12, y: 4.8 }, { id: "n4", x: 13, y: 5.1 },
  ];
  const newMid = showFit(store, "new", newPoints, "demo-new");
  const newModel = store.getModel(newMid);
  const byOld = projectPoints(oldModel, newPoints, 1);
  const byNew = projectPoints(newModel, newPoints, 1);
  console.log("  点     | 旧模型 score1 | 旧模型 err^2 | 重新拟合 score1 | 重新拟合 err^2");
  for (let i = 0; i < newPoints.length; i++) {
    console.log(`  ${newPoints[i].id.padEnd(6)}| ${byOld[i].score1.toFixed(4).padStart(13)} | ${byOld[i].err2.toExponential(3).padStart(12)} | ${byNew[i].score1.toFixed(4).padStart(15)} | ${byNew[i].err2.toExponential(3).padStart(12)}`);
  }
  console.log("  说明：旧模型均值固定在旧训练集上，新点整体平移后得分与误差显著变大；重新拟合会重算均值与主轴。");

  line("5. 独立核验（重启后从数据库重放仍复现）");
  for (const mid of [mid1, oldMid, newMid]) {
    const m = store.getModel(mid);
    const rep = verifyModel(m, store.getSamplePoints(m.sample_id));
    console.log(`  模型 #${mid}: ${rep.ok ? "全部通过" : "失败"}`);
    for (const c of rep.checks) console.log(`    [${c.ok ? "OK" : "FAIL"}] ${c.name}: ${c.detail}`);
  }
}
