# 552 Bank Marketing（Term Deposit Subscription Prediction）

本仓库用于完成 UBC MDS DATA 552 项目：基于葡萄牙银行直销营销数据，预测客户是否会订阅定期存款（`y=yes/no`）。包含 **EDA** 与可复现的 **baseline 建模评估流水线**（训练/测试划分、指标、Top-k 业务指标、输出图表）。

---

## 目录结构

```
.
├── bank/                 # 原始数据（bank 数据版本，解压后的文件夹）
├── bank-additional/      # 原始数据（bank-additional 数据版本，解压后的文件夹）
├── code/
│   ├── baseline_pipeline.py   # 主流水线脚本（训练/评估/产出图表）
│   └── eda.ipynb              # EDA Notebook
├── data/                 # 处理后的/中间数据（建议不提交到 git）
└── outputs/              # 运行产出（指标、figs 等，建议不提交到 git）
```

> 说明：`bank/` 与 `bank-additional/` 通常来自 UCI Bank Marketing 数据集的不同版本。你的脚本会使用其中某一个（或两者），以 `baseline_pipeline.py` 的实际读取路径为准。

---

## 快速开始

### 1) 创建环境（推荐：conda + pip）

```bash
conda create -n bank552 python=3.11 -y
conda activate bank552

pip install -U pip
pip install numpy pandas scikit-learn matplotlib joblib jupyter
```

如脚本使用了额外依赖（例如 `xgboost` / `lightgbm`），按脚本 import 安装即可。

---

## 数据准备

### 原始数据放置

- 将解压后的数据文件夹放在项目根目录下：
  - `bank/` 或 `bank-additional/`

### 中间数据与产出

- `data/`：中间数据（可选）
- `outputs/`：运行输出（图、指标文件等）

建议不要把 `data/` 与 `outputs/` 提交到 git（避免仓库过大，且便于复现）。

---

## 运行方式

在项目根目录运行：

```bash
python code/baseline_pipeline.py
```

一般会在终端输出：
- ROC-AUC、PR-AUC、F1 等分类指标
- Top 10% Precision / Lift 等业务指标

并把图表/结果保存到：
- `outputs/`（例如 `outputs/figs/`）

---

## 建模与评估约定

- **Target**：`y`（`yes`/`no`）
- **Class imbalance**：通常 `y=yes` 占比较低，因此 **PR-AUC** 与 **Top-k** 指标更有解释力
- **Deployable vs Upper-bound（如适用）**
  - 部署模型：排除泄露特征（常见：`duration`）
  - 上界模型：可包含 `duration` 作为性能上限参考（不用于真实部署）

---

## EDA（可选但推荐）

打开 notebook：

```bash
jupyter notebook code/eda.ipynb
```

建议在提交/协作前清理 notebook 输出（降低 diff 与仓库体积）：

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace code/eda.ipynb
```

---

## Git 建议（可选）

推荐 `.gitignore` 至少包含：

- `data/`
- `outputs/`
- `.ipynb_checkpoints/`
- `__pycache__/`

示例：

```gitignore
data/
outputs/
.ipynb_checkpoints/
__pycache__/
*.pyc
.DS_Store
```

---

## 复现说明

如需他人完全复现你的结果，请在以下位置补充/固定：
- `baseline_pipeline.py` 中的数据读取路径与文件名
- 随机种子（`random_state`）
- 训练/测试划分策略
- 输出路径（`outputs/`）

---

## 许可

课程项目用途。如需开源发布，请补充 License。
