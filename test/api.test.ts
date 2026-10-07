import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { Store } from "../src/store";
import { startServer } from "../src/server";
import { PcaError } from "../src/pca";

let store: Store;
let base: string;
let closeServer: () => Promise<void>;

before(async () => {
  store = new Store(":memory:");
  const { server, port } = await startServer(store, 0);
  base = `http://127.0.0.1:${port}`;
  closeServer = () => new Promise<void>((r) => server.close(() => r()));
});
after(async () => { await closeServer(); store.close(); });

async function api(method: string, path: string, body?: unknown): Promise<{ status: number; json: unknown }> {
  const res = await fetch(base + path, {
    method,
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return { status: res.status, json: await res.json() };
}

const PTS = { points: [{ id: "a", x: -1, y: 0 }, { id: "b", x: 0, y: 0 }, { id: "c", x: 1, y: 0 }] };

test("完整 API 流程：样本→拟合→主轴→投影→核验→导出", async () => {
  const s = await api("POST", "/samples", { name: "t1", ...PTS });
  assert.equal(s.status, 201);
  const sampleId = (s.json as { sampleId: number }).sampleId;

  const f = await api("POST", "/models", { sampleId });
  assert.equal(f.status, 201);
  const model = (f.json as { model: { id: number; lambda1: number; axis1: { x: number } } }).model;
  assert.ok(Math.abs(model.lambda1 - 1) < 1e-12);
  assert.ok(Math.abs(Math.abs(model.axis1.x) - 1) < 1e-12);

  const g = await api("GET", `/models/${model.id}`);
  assert.equal(g.status, 200);

  const p = await api("POST", `/models/${model.id}/project`, { train: true, k: 1, save: true });
  assert.equal(p.status, 200);
  const proj = p.json as unknown as { totalSquaredError: number; explainedVarianceRatio: number; rows: unknown[] };
  assert.ok(proj.totalSquaredError < 1e-24);
  assert.equal(proj.explainedVarianceRatio, 1);
  assert.equal(proj.rows.length, 3);

  const pe = await api("POST", `/models/${model.id}/project`, { k: 2, points: [{ id: "x", x: 5, y: 2 }] });
  assert.equal(pe.status, 200);

  const v = await api("GET", `/models/${model.id}/verify`);
  assert.equal((v.json as { report: { ok: boolean } }).report.ok, true);

  const e = await api("GET", `/models/${model.id}/export`);
  assert.equal(e.status, 200);
  const exp = e.json as unknown as { trainPoints: unknown[]; projections: unknown[]; verify: { ok: boolean } };
  assert.equal(exp.trainPoints.length, 3);
  assert.equal(exp.projections.length, 3, "保存的投影结果持久化");
  assert.ok(exp.verify.ok);
});

test("错误码：非法输入 400 / 未找到 404 / 未知路由 404", async () => {
  const bad = await api("POST", "/samples", { points: [{ id: "a", x: NaN, y: 0 }] });
  assert.equal(bad.status, 400);
  assert.equal((bad.json as { error: { code: string } }).error.code, "INVALID_INPUT");

  const one = await api("POST", "/samples", { points: [{ id: "a", x: 0, y: 0 }] });
  assert.equal(one.status, 400);

  const nf = await api("GET", "/models/9999");
  assert.equal(nf.status, 404);
  assert.equal((nf.json as { error: { code: string } }).error.code, "NOT_FOUND");

  const nr = await api("GET", "/nope");
  assert.equal(nr.status, 404);

  const bk = await api("POST", "/models/1/project", { k: 3, points: [{ id: "a", x: 0, y: 0 }, { id: "b", x: 1, y: 1 }] });
  assert.equal(bk.status, 400);
});

test("失败回滚：非法样本不产生记录，也不污染后续模型", async () => {
  const before = (await api("GET", "/samples")).json as unknown as { samples: unknown[] };
  const n0 = before.samples.length;
  await api("POST", "/samples", { name: "bad", points: [{ id: "a", x: 0, y: 0 }, { id: "b", x: Infinity, y: 1 }] });
  const afterJ = (await api("GET", "/samples")).json as unknown as { samples: unknown[] };
  assert.equal(afterJ.samples.length, n0, "失败请求不留下样本");
});

test("修改样本产生新模型版本，投影引用明确模型", async () => {
  const s1 = await api("POST", "/samples", { name: "v1", ...PTS });
  const id1 = (s1.json as { sampleId: number }).sampleId;
  const s2 = await api("POST", "/samples", { name: "v2", points: [...PTS.points, { id: "d", x: 10, y: 0 }] });
  const id2 = (s2.json as { sampleId: number }).sampleId;
  assert.notEqual(id1, id2);
  const m1 = (await api("POST", "/models", { sampleId: id1 })).json as { model: { id: number; lambda1: number } };
  const m2 = (await api("POST", "/models", { sampleId: id2 })).json as { model: { id: number; lambda1: number } };
  assert.notEqual(m1.model.id, m2.model.id);
  assert.ok(m2.model.lambda1 > m1.model.lambda1, "新样本改变模型");
});

test("Store 直接保存失败时回滚", () => {
  const s = new Store(":memory:");
  const sid = s.saveSample("ok", [{ id: "a", x: 0, y: 0 }, { id: "b", x: 1, y: 1 }]);
  assert.equal(s.listSamples().length, 1);
  assert.throws(() => s.getSamplePoints(999), (e: unknown) => e instanceof PcaError && (e as PcaError).code === "NOT_FOUND");
  // 人为制造写入失败：关闭数据库后保存应抛 DB_ERROR 且不留记录
  s.close();
  assert.throws(() => s.saveSample("x", [{ id: "a", x: 0, y: 0 }, { id: "b", x: 1, y: 1 }]), (e: unknown) => e instanceof PcaError && (e as PcaError).code === "DB_ERROR");
  void sid;
});


