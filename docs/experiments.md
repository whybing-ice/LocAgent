# 定位算法与实验复现

以下命令从仓库根目录运行。

## 二维 OLS 定位示例

运行 `uv run python` 后，可以用已知目标位置生成距离，再把锚点和距离交给 OLS 估计坐标。

```python
from locagent.algorithms.ols import estimate_position_ols
from locagent.simulation.data_generator import generate_noiseless_ranges

anchors = [(0, 0), (10, 0), (0, 10), (10, 10)]
ranges = generate_noiseless_ranges(anchors, (2, 3))
position = estimate_position_ols(anchors, ranges)
print(position)  # [2. 3.]
```

实测时，按锚点顺序传入距离，并让坐标和距离使用相同单位。至少需要三个锚点，且所有锚点不能共线。OLS 对平方距离方程做中心化，再用线性最小二乘估计坐标。

## 二维 WLS 定位示例

WLS 为每个锚点的平方距离方程指定一个权重。按锚点顺序传入有限正权重，权重越大，该方程对拟合的影响越大。权重全部相等时，结果与 OLS 一致。

```python
import numpy as np

from locagent.algorithms.ols import estimate_position_ols
from locagent.algorithms.wls import estimate_position_wls
from locagent.simulation.data_generator import generate_noiseless_ranges

anchors = [(0, 0), (10, 0), (0, 10), (10, 10)]
target = (2, 3)
ranges = generate_noiseless_ranges(anchors, target)
ranges[-1] += 4
ols = estimate_position_ols(anchors, ranges)
wls = estimate_position_wls(anchors, ranges, [1, 1, 1, 0.05])

ols_error = np.linalg.norm(ols - target)
wls_error = np.linalg.norm(wls - target)
print(f"OLS: {ols_error:.3f}, WLS: {wls_error:.3f}")
print(f"误差降幅：{1 - wls_error / ols_error:.2%}")
```

这个仿真预先知道最后一个测距有偏差，并将其权重设为 0.05。以估计坐标到真实坐标的欧氏距离衡量，误差从 3.572 降到 0.621，降幅为 82.61%。实际蓝牙测距的定位精度需要用实测数据评估。

## 用重复测距校准 WLS

`estimate_position_calibrated_wls(anchors, ranges, calibration_errors)` 接收按锚点排列的二维校准误差数组。每行至少包含两次在已知距离下得到的「实测距离减真实距离」，单位与坐标一致。它先减去各锚点的平均偏差，再按估计的平方距离方差给权重，最后调用原有 WLS 求解一次。默认测距标准差下限为 0.05，调用时可用 `min_range_std` 调整。

运行以下命令复现对比。实验用每个锚点的 100 条独立校准样本估计偏差和波动，并在同一批 1000 个仿真目标上评估两种方案。

```powershell
uv run python experiments/experiment_1_noise.py
```

| 方法 | 误差 ≤ 0.8 m | 中位误差 | 90% 分位误差 |
| --- | ---: | ---: | ---: |
| 等权 WLS | 45.9% | 0.853 m | 2.091 m |
| 校准 WLS | 98.7% | 0.289 m | 0.538 m |

该仿真假设各锚点的测距偏差和噪声水平在校准与定位期间保持稳定。真实场景的达标率需用独立实测数据验证。

## Huber 鲁棒定位

Huber 从 OLS 坐标出发，迭代拟合预测距离与实测距离之间的差。残差绝对值超过 `delta` 后，惩罚由平方增长转为线性增长。坐标以米计时，默认 `delta=0.5` 对应 0.5 m 的单条测距残差阈值。最终坐标误差仍按 0.8 m 评估。

```python
from locagent.algorithms.huber import estimate_position_huber
from locagent.simulation.data_generator import generate_noiseless_ranges

anchors = [(0, 0), (10, 0), (0, 10), (10, 10), (5, 0), (5, 10)]
ranges = generate_noiseless_ranges(anchors, (2, 3))
ranges[3] += 4
print(estimate_position_huber(anchors, ranges, delta=0.5))
```

```powershell
uv run python experiments/experiment_2_outliers.py
```

固定随机种子，在同一批 1,000 个目标上比较普通测距噪声与额外加入一条 4 m 正偏差的场景。异常锚点随机选取。原版 WLS 使用仿真设定的测距方差倒数作为固定权重，校准 WLS 使用独立生成的校准样本。

| 方法 | 普通噪声达标率 | 单条异常达标率 | 单条异常 90% 分位误差 |
| --- | ---: | ---: | ---: |
| OLS | 100.0% | 8.1% | 2.890 m |
| WLS | 100.0% | 4.2% | 3.162 m |
| 校准 WLS | 100.0% | 30.2% | 2.262 m |
| Huber | 100.0% | 98.3% | 0.580 m |

达标定义为二维位置误差不超过 0.8 m。以上结果来自仿真，真实蓝牙测距的达标率仍需独立实测。

## Agent 窗口选择实验

固定策略和 Agent 都使用 Huber。每个锚点保留窗口内最新的一条测距。Agent 用近期短窗口估计的坐标变化判断是否疑似移动，再检查是否至少有三个不共线的近期锚点，满足条件时使用短窗口。否则使用长窗口。日志记录窗口、锚点数量、选择理由和定位误差。

```powershell
uv run python experiments/experiment_3_agent.py
```

这条命令无需下载数据。移动仿真中，固定窗口与自适应窗口均出解 300/300 次，中位误差分别为 0.973 m 和 0.966 m，CEP90 分别为 2.046 m 和 2.248 m。

### 公开 BLE 轨迹回放

取得对应采集日的 BLE CSV 文件并放在仓库根目录的 data 文件夹，也可以在命令中填写文件的实际路径。以 6 月 26 日记录拟合 RSSI 到距离模型，6 月 28 日作为验证日，7 月 3 日用于第一轮跨日评估。固定窗口为 45 秒，Agent 可选择 15 秒窗口。机器人真值速度用于评估分组，在线决策使用历史定位估计。[Pettorru 等人的数据目录](https://github.com/GiovanniPettorru/multimodal-RSSI-localization-dataset/tree/main/RSSI%20Dataset)

```powershell
uv run python experiments/replay_public_ble.py data/26_06_BLE.csv data/28_06_BLE_1.csv --log results/tables/public_ble_validation.csv
uv run python experiments/replay_public_ble.py data/26_06_BLE.csv data/03_07_BLE.csv
```

以上命令从 LocAgent 仓库根目录运行，CSV 路径相对于该目录。公开测试日匹配了 13,941 条 BLE 记录，15 秒与 45 秒窗口内的锚点数中位数为 9 和 13。

| 数据 | 场景 | 固定窗口 CEP90 | Agent CEP90 | 两种方法的出解数 |
| --- | --- | ---: | ---: | ---: |
| 6 月 28 日验证 | 静止 | 6.054 m | 5.467 m | 95/95 |
| 6 月 28 日验证 | 移动 | 23.928 m | 20.279 m | 1524/1525 |
| 7 月 3 日测试 | 静止 | 10.288 m | 10.462 m | 658/659 |
| 7 月 3 日测试 | 移动 | 6.626 m | 7.252 m | 6660/6660 |

CEP90 按成功出解时刻计算。本次 Agent 在验证日改善了移动误差，测试日的移动 CEP90 则上升了约 9.4%。公开数据使用固定信标，表中统计涵盖所有出解的锚点数量，华为原题的正式验收需要按指定数据、上报设备数量和基线另行计算。

### 7 月 5 日独立评估

在查看该日结果前，固定了 15 秒、45 秒和原 Agent 三种策略。使用 6 月 26 日数据校准，运行以下命令复现。

```powershell
uv run python experiments/replay_public_ble.py data/26_06_BLE.csv data/05_07_BLE.csv --log results/tables/public_ble_holdout.csv
```

| 策略 | 移动 CEP90 | 移动出解数 | 静止 CEP90 | 静止出解数 |
| --- | ---: | ---: | ---: | ---: |
| 固定 15 秒 | 4.544 m | 944/947 | 5.998 m | 120/122 |
| 固定 45 秒 | 4.415 m | 947/947 | 5.773 m | 122/122 |
| 原 Agent | 4.598 m | 947/947 | 5.902 m | 122/122 |

固定 45 秒在该日的移动和静止场景中均取得最低 CEP90。该结果与 6 月 28 日对固定长窗的比较方向不同，当前 Agent 的收益尚未跨日复现。

## 大模型辅助离线检验

脚本先在验证日回放固定长窗口、固定短窗口和当前 Agent，只把各组的 CEP90、出解数和总数发给 DeepSeek。模型提出连续多少次满足运动条件后才切换窗口，允许 2 至 5 次。Python 复跑同一批观测，输出候选策略的实际得分。运行后会在 `results/tables/llm_validation/` 保存提案、各时刻决策和评分。实时定位可独立运行。

在 PowerShell 进程中配置 `DEEPSEEK_API_KEY` 后运行以下命令。使用 6 月 28 日验证数据提出和比较候选策略。

```powershell
uv run python experiments/validate_llm_strategy.py data/26_06_BLE.csv data/28_06_BLE_1.csv
```

使用同济学校网关签发的密钥时，在同一个 PowerShell 会话中指定学校地址和模型。脚本会在给定地址后追加 `/chat/completions`。

先检查接口能否返回一条模型消息。此命令无需 BLE 数据。

```powershell
$env:DEEPSEEK_API_KEY = (Read-Host "粘贴学校 API 密钥" -MaskInput).Trim()
uv run python experiments/validate_llm_strategy.py --check-api --base-url 'https://agi.tongji.edu.cn/v1' --model 'DeepSeek-V4.1-Flash'
```

```powershell
$dataDir = 'data'
uv run python experiments/validate_llm_strategy.py (Join-Path $dataDir '26_06_BLE.csv') (Join-Path $dataDir '28_06_BLE_1.csv') --base-url 'https://agi.tongji.edu.cn/v1' --model 'DeepSeek-V4.1-Flash' --output-dir results/tables/tongji_live
```

已有提案可通过 `--proposal 路径` 在本地复跑。提案文件格式示例为 `{"motion_confirmations": 2, "hypothesis": "连续两次判断可减少偶发抖动造成的切换"}`。锁定候选规则后，在独立数据上用 `replay_public_ble.py --motion-confirmations 2` 执行最终评估，数字以实际选定规则为准。

[交付清单](../TODO.md)

## 筛选疑似异常坐标上报

给 `locate` 传入 `rejection_threshold=3` 可启用筛选。窗口内至少有五条来自不同锚点的上报时，逐个留出并重算。在返回的 `decision.assessments` 中，每份记录含设备 ID、上报时间、状态、原因、留出残差和其余上报平均绝对残差的下降量。默认关闭筛选时，该集合为空。

| 状态 | 判断条件 |
| --- | --- |
| `normal` 正常使用 | 留出该上报后，它与重算位置的测距残差小于 3 m。 |
| `suspect_kept` 可疑但保留 | 残差达到 3 m，同时出现多条冲突、整体筛查不完整，或其余上报的改善小于 0.03 m。 |
| `rejected` 本次剔除 | 完成筛查后只发现一条冲突，并且其余上报的改善达到 0.03 m。 |

如果锚点不足或某次求解无法完成，对应状态为 `None`，原因字段记录未能评估的条件。筛选只对当前窗口生效。

```powershell
uv run python experiments/experiment_4_report_filter.py
uv run python experiments/experiment_4_report_filter.py --seed 43
uv run python experiments/experiment_4_report_filter.py --seed 44 --log results/tables/report_screening_seed44.csv
```

实验在固定窗口内让两组都使用 Huber。每组 300 次定位，模拟 12 台设备围绕目标，保留依据设备真实位置生成的测距，只给一台设备的上报坐标施加随机方向的偏移。输出包括剔除数、出解数、CEP90，以及异常和正常上报各自的标记数与总数。标记数包含可疑保留和实际剔除。使用 `--log` 保存逐上报明细，注入真值仅用于评分。
