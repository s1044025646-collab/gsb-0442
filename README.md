# PCA2D Backend — 二维 PCA 主轴投影与重建误差核验

纯本地后端（TypeScript + Node.js + SQLite），只做 API 与 CLI，无 Docker / WSL / 外部服务依赖。
对带编号的二维 (x, y) 样本做 PCA：均值、样本协方差、两条正交主轴与特征值，支持保留 1 或 2 个主成分进行投影与重建，并提供独立核验入口。

## 环境要求

- Windows 本地，Node.js ≥ 22.5（使用内置 `node:sqlite`，无需原生编译；开发使用 Node 25 验证）
- 仅开发依赖：`typescript`、`@types/node`、`tsx`（`npm install` 安装）

## 快速开始

```powershell
npm install
npm run build     # 编译 TypeScript 到 dist/
npm test          # 构建并运行全部测试（node:test）
npm start         # 启动 API（默认自动选择空闲端口；--port 8080 或 PORT 环境变量指定）
npm run cli -- fit demo "[{\"id\":\"a\",\"x\":-1,\"y\":0},{\"id\":\"b\",\"x\":0,\"y\":0},{\"id\":\"c\",\"x\":1,\"y\":0}]"
npm run demo      # 内置演示：水平三点 / 斜线 / 各向同性四点 / 旧模型 vs 重新拟合 / 核验
```

数据目录为项目内 `data/`（SQLite 文件 `data/pca2d.db`），重启后历史数据集、模型与投影结果均可复现。

## 数值口径（重要）

- **均值**：由全部输入点计算；投影新点时**始终使用训练均值**，绝不用新点重新拟合。
- **协方差**：先中心化再外积累加，分母为 **n-1**（样本协方差），不使用未去均值的二阶矩。
- **特征分解**：自行闭式求解 2×2 对称矩阵（迹 ± 判别式平方根），按特征值降序；不调用任何现成 PCA 训练器。
- **符号规则**：特征向量整体符号有二义性，固定为“第一个绝对值显著非零（>1e-12）的分量取正”。
- **尺度**：**不**按坐标标准差缩放。x、y 单位不同会直接影响主轴方向，报告中保留输入单位与原始坐标。
- **解释方差比例**：分母为两个特征值之和（λ1+λ2），不是轴向量长度（轴恒为单位向量）。
- **得分**：(新点 − 训练均值) 与各主轴的内积；**重建**：保留得分 × 主轴 + 原均值。两轴是方向不是平移向量。
- **容差**：公开固定值 `1e-10`（`src/pca.ts` 的 `DEFAULT_TOLERANCE`，拟合时可覆盖），用于极小负特征值归零、退化判定。

## 退化情形

- **全体相同 / 零方差**：采用公开固定坐标轴 e1=(1,0)、e2=(0,1)，特征值为 0；解释方差比例返回 `null` 并附 `zero_total_variance` 说明，不产生 NaN。
- **相等特征值（各向同性）**：主轴不唯一，报告附 `equal_eigenvalues` 说明并使用确定基 e1/e2。
- **近共线**：λ2 接近 0 但保持非负；极小负特征值（|λ| ≤ tol×scale）按容差归零，明显负值报 `NEGATIVE_EIGENVALUE` 错误，不掩盖算法错误。
- 允许重合点、共线点；至少 2 个点（外部投影点至少 1 个）。

## 输入限制与错误码

- 样本数 2 ~ 100,000；坐标绝对值 ≤ 1e12；拒绝缺失或非有限（NaN/Infinity）数值；点编号不得重复。
- 错误以 JSON 返回：`{"error": {"code": "...", "message": "..."}}`，HTTP 400（内部错误 500）。
- 错误码：`INVALID_INPUT` / `INVALID_JSON` / `INVALID_K` / `TOO_FEW_POINTS` / `TOO_MANY_POINTS` / `DUPLICATE_ID` / `NON_FINITE_VALUE` / `COORD_OUT_OF_RANGE` / `DATASET_NOT_FOUND` / `MODEL_NOT_FOUND` / `NEGATIVE_EIGENVALUE` / `EIG_DECOMPOSITION_FAILED` / `STORAGE_FAILED` / `NOT_FOUND`。
- 所有写库操作在事务中执行，失败即回滚：非法输入或保存失败不会残留半成品，也不会混用上一个模型的结果。

## HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/datasets` | 保存样本版本 `{name, points:[{id,x,y}]}` → `{datasetId}` |
| GET | `/datasets` | 列出数据集 |
| POST | `/models` | 拟合 `{datasetId, tolerance?}` → `{modelId, model}`（均值/协方差/主轴/特征值） |
| GET | `/models` | 列出模型 |
| GET | `/models/:id` | 查询主轴与解释方差比例 |
| POST | `/models/:id/project` | 投影 `{points?, k?}`；缺省投影该模型的训练点，k∈{1,2}；返回得分、重建坐标、逐点平方误差、总平方误差、解释方差比例，结果落库 |
| POST | `/models/:id/reconstruct` | 由得分重建 `{scores:[[s1,s2?],...]}` |
| GET | `/models/:id/verify` | 独立核验：主轴单位长度、正交性、C·v=λ·v 残差、k=2 训练点重建误差 |
| GET | `/models/:id/report` | 导出完整报告（模型、训练点、核验、历史投影） |

修改样本需新建数据集（`POST /datasets`），拟合产生新模型版本；投影始终引用明确的 `modelId`。

## CLI

```
node dist/src/cli.js fit <name> <pointsJSON|file>            保存样本并拟合
node dist/src/cli.js axes <modelId>                          查询主轴
node dist/src/cli.js project <modelId> [pointsJSON|file] [k] 投影（缺省训练点，k 缺省 2）
node dist/src/cli.js reconstruct <modelId> "[[1,2],[3]]"     由得分重建
node dist/src/cli.js verify <modelId>                        独立核验
node dist/src/cli.js report <modelId>                        导出报告 JSON
node dist/src/cli.js list                                    列出数据集与模型
node dist/src/cli.js demo                                    内置演示
```

CLI 输出均值、协方差矩阵、主轴、得分表与逐点误差表，不做绘图。

## 验证方式

- 测试（`test/pca.test.ts`，16 项）：已知三点例（均值原点、协方差 diag(1,0)、主轴沿 x、k=1 无损）、整体平移不变性、点顺序无关、全体相同、各向同性四点、斜线 45°、不同尺度影响主轴、近共线、k=2 全恢复、k=1 训练总平方误差 = (n−1)·λ2、投影不改模型、核验入口、负特征值报错、输入校验、重启复现、失败回滚、样本修改产生新版本。
- 协方差由中心化外积直接手算，投影得分用内积独立核对，不以编码/解码互相通过代替正确性。
- 实测：`npm test` 16/16 通过；API 冒烟（建库→拟合→投影→重建→核验→导出→非法输入→回滚）全部符合预期。

## 限制

- 仅二维、仅完整数值数据；不做分类、聚类、异常点剔除、标准化或真实预测。
- 不追求极端病态矩阵的无限精度；坐标量级差异过大（>1e12）会被拒绝。
- `node:sqlite` 在 Node 22/23 为实验特性（启动时有 ExperimentalWarning，不影响功能）。
- 单进程服务，无鉴权，面向本地使用。
