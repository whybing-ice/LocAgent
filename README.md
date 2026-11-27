# LocAgent

LocAgent 是一个二维蓝牙低功耗（BLE）定位研究项目。设想一件遗失物持续广播蓝牙信号，附近的手机上报自己的坐标、扫描时间和信号强度（RSSI）。LocAgent 用这些上报估计物品位置，并研究两个问题。

- **多早的上报还值得用？** 物品可能已经离开几秒前的位置，但只保留最新观测，又可能凑不齐定位所需的设备。
- **某台手机上报的位置值得信任吗？** 当一条上报与其他设备矛盾时，Agent 需要决定是带着疑点继续使用，还是在本次定位中剔除它。

项目用公开 BLE 轨迹检验第一个问题，用可控的坐标偏移仿真检验第二个问题。在独立采集日，固定长窗口取得了最低的移动定位误差。在单台设备坐标偏移 10 m 的仿真中，筛查规则识别出 78.3% 的异常上报。下文给出了这两项结果的对照和适用条件。

## 快速开始

使用 Python 3.13 和 [uv](https://docs.astral.sh/uv/)，在仓库根目录执行以下命令。首次安装需要下载依赖。下列仿真直接生成观测数据。

```powershell
uv sync --locked
uv run --locked pytest -q
uv run --locked python experiments/experiment_2_outliers.py
```

最后一条命令将同一批观测交给 OLS、WLS、校准 WLS 和 Huber，打印位置误差不超过 0.8 m 的比例。在某条测距被随机加大 4 m 的场景下，可以复现 Huber 的 98.3% 达标率。

## Agent 如何决策

- **选择窗口。** 回放脚本根据最近的位置估计判断运动状态。疑似移动且短窗内有至少三个不共线锚点时，Agent 采用短窗口，否则尝试长窗口。每台设备只取窗口内最新的一条上报。
- **可选的上报筛查。** 对每台设备暂时留出其上报并重新定位，记录测距冲突和其余设备的一致性变化。筛查区分正常使用、可疑但保留和本次剔除。证据不足时保留可疑上报。
- **定位。** 使用选定观测和 Huber 损失估计二维坐标。DeepSeek 参与离线实验，可以从验证数据的摘要提出候选规则，再由本地回放评分。

例如，某台手机自报坐标后，RSSI 对应的测距是 5 m。暂时拿掉这台手机，如果其余设备估计的目标位置距离它的上报坐标有 17 m，这条上报就产生了 12 m 的冲突。在筛查实验中，冲突达到 3 m 即标记为可疑。若它是唯一冲突，且留出它后其余设备的平均绝对残差至少下降 0.03 m，Agent 才执行剔除。记录里可以查到设备 ID、状态和理由。

下面的例子可以在项目环境中直接运行。前五台设备的观测与目标位置 `(3, 4)` 一致，最后一台设备的上报坐标被改成了 `(20, 4)`。

```python
from locagent.agent.loc_agent import RangeReport, locate

anchors = [(0, 0), (6, 0), (0, 8), (6, 8), (3, 9), (20, 4)]
reports = [RangeReport(0, f"phone-{i}", xy, 5.0) for i, xy in enumerate(anchors)]
decision = locate(reports, now=0, delta=1, rejection_threshold=3)

print(tuple(round(float(v), 2) for v in decision.position))  # (3.0, 4.0)
print(decision.rejected_anchor_id)                           # phone-5
```

## 运行实验

| 脚本（位于 `experiments/`） | 验证内容 | 输入 |
| --- | --- | --- |
| `experiment_1_noise.py` | 等权 WLS 和校准 WLS | 仿真 |
| `experiment_2_outliers.py` | 异常测距下的四种定位方法 | 仿真 |
| `experiment_3_agent.py` | 固定窗口与 Agent 窗口选择 | 仿真 |
| `experiment_4_report_filter.py` | 注入坐标偏差后的识别、误标和剔除效果 | 仿真 |
| `replay_public_ble.py` | 公开 BLE 轨迹上的 CEP90 和出解数 | 两个不同采集日的 CSV |
| `validate_llm_strategy.py` | 模型提案与原规则的定位效果 | 校准和验证 CSV，加提案文件或 API 密钥 |

## 数据与复现

公开回放使用 Pettorru 等人的室内 BLE 数据，包括固定信标坐标、带时间戳的 RSSI，以及机器人的位置和速度真值。位置真值用于校准和评分，速度真值用于把评估时刻划分为静止和移动。在线判断运动时，程序使用当时已经得到的定位估计。[公开数据与字段说明](https://github.com/GiovanniPettorru/multimodal-RSSI-localization-dataset)

| 采集日期（2024 年） | 文件 | 在项目中的用途 |
| --- | --- | --- |
| 6 月 26 日 | `26_06_BLE.csv` | 拟合 RSSI 到距离的换算模型 |
| 6 月 28 日 | `28_06_BLE_1.csv` | 验证窗口策略和 DeepSeek 提案 |
| 7 月 3 日 | `03_07_BLE.csv` | 首轮跨日评估 |
| 7 月 5 日 | `05_07_BLE.csv` | 在规则锁定后，独立比较三种窗口策略 |

按采集日期划分，可以让同一段轨迹中相邻的扫描留在同一组。将所需文件下载到仓库根目录的 `data/` 后，以下命令分别复跑独立评估和已保存的模型提案，日志写入 `results/tables/`。两条命令均无需 API 密钥。[原始 CSV 目录](https://github.com/GiovanniPettorru/multimodal-RSSI-localization-dataset/tree/main/RSSI%20Dataset)

```powershell
uv run python experiments/replay_public_ble.py data/26_06_BLE.csv data/05_07_BLE.csv --log results/tables/public_ble_holdout.csv
uv run python experiments/validate_llm_strategy.py data/26_06_BLE.csv data/28_06_BLE_1.csv --proposal docs/deepseek_proposal.json
```

## 实验结果

### 窗口策略能否跨日取得收益？

CEP90 为成功出解时定位误差的 90% 分位数，数值越小，说明九成定位结果落在更小的误差范围内。出解数用于检查策略是否在困难时刻也能给出坐标。

| 7 月 5 日独立评估 | 移动 CEP90 | 移动出解数 | 静止 CEP90 | 静止出解数 |
| --- | ---: | ---: | ---: | ---: |
| 固定 15 秒 | 4.544 m | 944/947 | 5.998 m | 120/122 |
| 固定 45 秒 | 4.415 m | 947/947 | 5.773 m | 122/122 |
| Agent 自适应窗口 | 4.598 m | 947/947 | 5.902 m | 122/122 |

6 月 28 日，Agent 将移动 CEP90 从固定长窗的 23.928 m 降至 20.279 m。在 7 月 5 日，固定长窗和 Agent 都覆盖了所有评估时刻，固定长窗的误差更小。当前规则的收益随采集日变化。

### 标记可疑上报后，要剔除多少条？

坐标筛查实验每次模拟 12 台设备，以真实坐标生成测距，随机改变其中一台的上报坐标。下表使用锁定规则后的随机种子 44，每组 300 次。两列 CEP90 使用相同的观测和 Huber 求解器。

| 坐标偏移 | 异常上报被标记 | 其中保留 / 剔除 | 正常上报误标 | 仅 Huber / 筛查后 CEP90 |
| --- | ---: | ---: | ---: | ---: |
| 0 m | 无注入 | — | 0/3600 | 0.317 / 0.317 m |
| 5 m | 179/300 | 114 / 65 | 0/3300 | 0.423 / 0.407 m |
| 10 m | 235/300 | 152 / 83 | 0/3300 | 0.421 / 0.407 m |
| 20 m | 269/300 | 184 / 85 | 0/3300 | 0.420 / 0.392 m |

有些坐标偏移对手机与目标之间的距离影响很小，仅凭单次测距很难发现。实验验证了单台设备坐标异常时的筛查效果，多台设备同时报错及真实手机上报的误标率仍待评估。

### DeepSeek 的提案效果如何？

通过同济网关，DeepSeek-V4.1-Flash 提议连续两次检测到运动后才切换到短窗口，希望减少定位抖动导致的误切换。在 6 月 28 日验证集上，移动 CEP90 从原规则的 20.279 m 上升至 20.443 m，两者都在 1,525 个移动时刻中出解 1,524 次。独立评估沿用了原规则。[保存的模型提案](docs/deepseek_proposal.json)

算法与 Agent 代码在 `src/locagent/`，测试在 `tests/`。详细示例、参数、API 配置和分组结果见[实验复现指南](docs/experiments.md)和[课程报告提纲](docs/report_outline.md)。
