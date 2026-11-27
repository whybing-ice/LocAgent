# LocAgent

LocAgent 是一个二维蓝牙测距定位课程项目。计划用多部位置已知的手机提供的距离观测，估计目标物品的二维坐标，并研究 Agent 能否根据观测质量选择合适的定位方法。

**当前进度**：仓库骨架。仿真、定位算法、Agent、评估、绘图和演示界面均为 TODO。

## 1. 准备环境

在 Windows PowerShell 中进入包含本文件的 LocAgent 文件夹。项目使用 Python 3.13 和 uv。

```powershell
uv sync
uv run pytest
```

`uv sync` 会按项目配置准备虚拟环境并安装依赖。首次解析依赖时会生成 `uv.lock`，生成后可以和项目文件一起提交。当前测试只检查项目包能否导入。

## 2. 认识目录

| 位置 | 计划用途 |
| --- | --- |
| `src/locagent/simulation/` | 生成手机坐标、真实距离、带噪距离和异常观测 |
| `src/locagent/algorithms/` | 普通最小二乘（OLS）、加权最小二乘（WLS）、Huber 鲁棒定位 |
| `src/locagent/agent/` | 分析观测、选择策略、编排定位工具 |
| `src/locagent/evaluation/` | 统计定位误差、误差标准差和运行时间 |
| `src/locagent/visualization/` | 绘制二维场景和实验对比图 |
| `experiments/` | 比较噪声、异常值和 Agent 自适应策略 |
| `tests/` | 包导入检查，后续添加仿真和定位测试 |
| `configs/` | 实验参数示例 |
| `docs/` | 课程报告提纲 |
| `app/` | 可选的演示界面 |
| `notebooks/`、`results/` | 探索分析及实验产物 |

## 3. 从仿真开始

先实现 `data_generator.py`。给定手机的二维坐标和目标真实位置，先计算欧氏距离，再逐步加入噪声、不同测距精度和异常值。使用固定随机种子复现实验。目标真实位置用于生成数据和评价误差。

## 4. 提交到 GitHub

在 GitHub 创建空仓库 `LocAgent`。创建时让 README、许可证和 .gitignore 保持未选中。再在本地项目目录执行以下命令，把 `YOUR-ACCOUNT` 换成 GitHub 用户名或组织名。

```powershell
git status --short
git add .
git diff --cached --name-only
git commit -m "Set up LocAgent project skeleton"
git remote add origin https://github.com/YOUR-ACCOUNT/LocAgent.git
git push -u origin main
```

如果 Git 提示缺少提交者信息，先用 `git config user.name "你的名字"` 和 `git config user.email "你的提交邮箱"` 为这个仓库设置身份，再重试提交。`.env.example` 是可提交的模板。将来如需填密钥，请使用已被 Git 忽略的本地 `.env`。
