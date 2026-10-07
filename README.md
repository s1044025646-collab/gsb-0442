# 二维 PCA 主轴投影与重建误差核验后端

纯本地运行的二维 PCA 服务：TypeScript + Node.js + SQLite（`node:sqlite`，无原生编译依赖），只提供 HTTP API 与 CLI，不含绘图与前端。

## 功能

- 输入带编号的二维点 `(id, x, y)`，计算均值、样本协方差（分母 `n-1`）、两条正交单位主轴与特征值（降序）。
- 保留 1 或 2 个主成分进行投影（得分）与重建，输出逐点平方误差、总平方误差与解释方差比例。
- 独立核验入口：主轴正交归一、`C·v = λ·v` 残差、谱分解重建 `C ≈ Σλᵢvᵢvᵢᵀ`、训练集重建误差恒等式。
- SQLite 持久化样本版本、模型（均值/协方差/主轴/容差/说明）与投影结果；修改样本产生新模型，投影引用明确的模型 ID；重启后历史报告可复现。
- 多步写入全部在事务内，失败整体回滚，不混用上一个模型的结果。

## 环境要求

- Windows 本地，Node.js ≥ 22.5（开发使用 Node 25，`node:sqlite` 为实验特性，启动时有一条 ExperimentalWarning，属正常）。
- 不需要 Docker / WSL / 任何外部服务或在线 API。

## 构建、测试、启动、演示

```powershell
npm install        # 仅安装 typescript 与 @types/node（开发依赖）
npm run build      # 编译到 dist/
npm test           # 构建并运行全部单元测试与 API 测试
npm start          # 启动 HTTP API（默认自动选择空闲端口，打印实际端口）
npm run demo       # 内置演示：已知例 / 斜线 / 各向同性 / 旧模型 vs 重新拟合 / 核验
```

环境变量：`PCA_PORT` 指定端口（默认 0 = 空闲端口），`PCA_DB` 指定数据库路径（默认 `./data/pca.db`）。

## CLI 用法

```powershell
node dist/src/cli.js serve --port 8080      # 启动 API
node dist/src/cli.js add-sample 名称 points.json   # 保存样本新版本
node dist/src/cli.js samples                # 列出样本
node dist/src/cli.js fit <sampleId>         # 拟合并保存模型
node dist/src/cli.js models                 # 列出模型
node dist/src/cli.js axes <modelId>         # 查看均值/协方差/主轴/特征值
node dist/src/cli.js project <modelId> --k 1 --train        # 投影训练点
node dist/src/cli.js project <modelId> --k 2 --file new.json --save
node dist/src/cli.js verify <modelId>       # 独立核验
node dist/src/cli.js export <modelId> out.json             # 导出完整报告
node dist/src/cli.js demo                   # 演示
```

点文件格式：`[{"id":"a","x":-1,"y":0}, ...]` 或 `{"points":[...]}`。

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/samples` | 保存样本版本，body `{name?, points:[{id,x,y}]}` |
| GET | `/samples` | 列出样本 |
| POST | `/models` | 拟合模型，body `{sampleId}` |
| GET | `/models` / `/models/:id` | 列出 / 查询主轴等 |
| POST | `/models/:id/project` | 投影，body `{k:1\|2, train:true}` 或 `{k, points:[...], save?}` |
| GET | `/models/:id/verify` | 独立核验分解与重建误差 |
| GET | `/models/:id/export` | 导出模型+训练点+已存投影+核验报告 |

错误统一为 `{"error":{"code","message"}}`：`INVALID_INPUT`(400)、`NOT_FOUND`(404)、`NUMERIC_FAILURE`(422)、`DB_ERROR`(500)。

## 算法口径（重要假设）

- **中心化**：均值由全部训练点计算；新点投影时减的是训练均值，绝不用新点重新拟合。
- **协方差**：先中心化再累加外积，分母 `n-1`（样本协方差），不使用未去均值的二阶矩。
- **特征分解**：自行解析求解 2×2 对称矩阵（`λ = tr/2 ± hypot((a-c)/2, b)`），特征向量取 `(b, λ-a)` 与 `(λ-c, b)` 中范数较大者并单位化；第二轴取第一轴的垂直方向保证严格正交。不调用任何现成 PCA 库。
- **符号规则**：特征向量整体符号有二义性，统一取“第一个绝对值显著非零的分量为正”。
- **解释方差比例**：`λᵢ / (λ₁+λ₂)`，分母是两个特征值之和，不是轴向量长度（轴恒为单位向量）。
- **重建**：`均值 + Σ(保留得分 × 主轴)`；两轴不会作为平移向量相加。
- **不缩放**：不按每个坐标的标准差缩放，坐标保留输入单位；尺度差异会影响主轴方向，报告中明确注明。

## 退化与数值处理

- **全零方差**（所有点重合）：采用公开固定坐标轴 `e1=(1,0)`、`e2=(0,1)`，解释方差比例按“无方差”口径返回 `[0,0]`，不产生 NaN。
- **重复特征值**（如各向同性四点）：主轴方向不唯一，报告注明并使用确定基。
- **极小负特征值**：相对容差 `1e-9` 内按数值误差归零；明显为负或分解残差异常返回 `NUMERIC_FAILURE`，不掩盖算法错误。
- **输入限制**：至少 2 个点（投影外部点至少 1 个），样本数 ≤ 100000，坐标绝对值 ≤ 1e12，拒绝缺失、NaN、Infinity 与重复 id；不追求极端病态矩阵的无限精度。

## 验证结论（开发时实际运行）

- `npm test`：18 个测试全部通过，覆盖已知例 `(-1,0),(0,0),(1,0)`（均值原点、协方差 diag(1,0)、保留一维无损）、整体平移、点顺序、全重合、近共线、不同尺度、斜线 45°、各向同性重复特征值、`SSE(k=1) = (n-1)·λ₂` 恒等式、独立内积核对投影、非法输入拒绝、API 全流程、失败回滚。
- `npm run demo`：展示主轴旋转、重复特征值限制、旧模型投影新点与重新拟合的差别，核验全部通过；重启进程后对持久化模型再次 `verify` 结果一致。

## 限制

- 仅二维、只做完整数值数据；不做分类、聚类、异常点剔除、真实预测。
- `node:sqlite` 目前标记为实验特性，未来 Node 版本 API 可能微调。
- 浮点结果受双精度限制，核验使用公开相对容差而非逐位相等。
