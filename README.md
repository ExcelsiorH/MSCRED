# Pytorch-MSCRED

# MSCRED — 多尺度卷积自编码时序异常检测（项目说明）

**项目概述**

- **简介**: 本仓库实现了基于 MSCRED（Multi-Scale Convolutional Recurrent Encoder-Decoder）的时序异常检测流程。通过将多通道时序数据转换为矩阵，使用卷积自编码器 + ConvLSTM 做重建，再基于重建误差进行异常检测与根因分析。

**环境依赖**

- **Python/包**: 推荐使用 conda 环境，仓库提供 `environment.yml` 和 `requirements.txt`，可选其一来安装依赖。

  - 使用 conda (推荐): `conda env create -f environment.yml`
  - 使用 pip: `python -m pip install -r requirements.txt`

**项目文件说明（重要）**

- **`main.py`**: 训练与测试入口脚本。依赖 `utils.data.load_data()` 返回的 `train`/`test` DataLoader。训练结果会保存到 `./checkpoints`，测试阶段会把重建结果写到 `data/matrix_data/reconstructed_data/`。
- **`visualization_app.py`**: 基于 Streamlit 的可视化界面，用于绘制异常分数曲线、阈值、热力图及简单的根因候选传感器提示。运行命令：`streamlit run visualization_app.py`。
- **`evaluate.ipynb`**: 交互式评估笔记本（用于对检测结果进行可视化/评估）。

**目录说明**

- **`data/`**: 数据目录（原始数据、处理后数据、矩阵形式的数据与重建数据）

  - **`origin_data/`**: 原始按日期组织的 CSV 源文件（按采集批次/日期分文件夹）。
  - **`train_raw/`、`test_raw/`**: 合并后的原始 CSV（未归一化/未切片）。
  - **`train_processed/`、`test_processed/`**: 经过预处理/切片/归一化后的 CSV（可直接用于构建矩阵）。
  - **`matrix_data/`**: 训练/测试的矩阵样本目录（包含 `train_data/`、`test_data/`、`reconstructed_data/`）。
- **`model/`**: 模型实现

  - **`mscred.py`**: MSCRED 模型主体（包含 Encoder/ConvLSTM/Decoder 与 attention 逻辑）。
  - **`convolution_lstm.py`**: ConvLSTM 与其 Cell 的实现（时序特征提取核心）。
- **`utils/`**: 项目中活跃的预处理与数据生成脚本

  - **`0_merge_original_data.py`**: 合并原始传感器 CSV 并按工序时间段切片（示例脚本，需按环境修改路径与工序时间表）。
  - **`1_stage_normalize_multi_file.py`**: 阶段性归一化/标准化处理脚本（用于把合并后的 CSV 转为模型输入范围）。
  - **`data.py`**: 数据加载器（实现 `load_data()`，负责把 CSV/矩阵封装成 PyTorch DataLoader）。
  - **`matrix_generator_multi.py`**: 将时间序列切成矩阵样本（滑动窗口、步长、通道组织），输出到 `data/matrix_data/`。
- **`checkpoints/`**: 模型权重保存目录（例如 `model0928.pth`、`model2.pth`，以及训练脚本保存的 `model_1121.pth`）。
- **`outputs/`**: 结果输出（如 `error_heatmaps/`，以及其它分析/评估产物）。
- **`used_utils/`**: 历史/废弃脚本目录（包含早期实验代码与遗留脚本）。注意：除非特别需要，否则请不要使用 `used_utils/` 内的脚本——这些脚本已标为废弃/历史版本，仓库主线的 `utils/` 才是当前首选。

**典型使用流程（推荐顺序）**

1. **准备环境**

- 创建 conda 环境（或使用 pip 安装）：

```powershell
conda env create -f environment.yml
# 或者
python -m pip install -r requirements.txt
```

2. **准备原始数据**

- 将原始传感器 CSV 放入 `data/origin_data/<batch_date>/`（参考 `utils/0_merge_original_data.py` 中的路径配置与 `PROCESS_SCHEDULE`）。

3. **合并与预处理**

- 使用 `utils/0_merge_original_data.py` 按工序时间段合并并输出到 `data/*_raw/`。
- 使用 `utils/1_stage_normalize_multi_file.py` 等脚本对合并后的 CSV 做归一化/标准化，生成 `data/*_processed/`。

4. **构建矩阵样本**

- 运行 `utils/matrix_generator_multi.py`（或相应脚本）把时序切成形状为 `(Step, C, H, W)` 的矩阵样本，输出到 `data/matrix_data/train_data/` 与 `data/matrix_data/test_data/`。

5. **训练模型**

- 确认 `utils/data.py` 中的 `load_data()` 正确加载 `matrix_data` 下的样本并返回 `{'train': train_loader, 'test': test_loader}`。
- 运行训练（默认在 `main.py` 中）:

```powershell
python main.py
```

- 训练结束后会把模型参数保存到 `./checkpoints/`。

6. **推理/测试**

- `main.py` 中测试流程会把重建矩阵写入 `data/matrix_data/reconstructed_data/`，可用于后续的阈值判定与可视化。

7. **可视化与评估**

- 启动可视化面板：

```powershell
streamlit run visualization_app.py
```

- 或打开 `evaluate.ipynb` 进行离线评估与绘图。

**重要说明与建议**

- **不要使用 `used_utils/` 中已废弃脚本**，优先参考 `utils/` 下的脚本与 `main.py` 的数据路径配置。
- 在运行任何预处理脚本前，请检查脚本顶部的路径常量（示例脚本常包含硬编码路径）。建议将路径修改为你的本地工作路径或改写为命令行参数。
- `visualization_app.py` 默认依赖 `data/test_anomaly.csv`（用于 ground-truth），请确保该文件存在或适当修改脚本读取路径。
- 若你的 GPU 可用，训练脚本会自动使用 CUDA；如需强制使用 CPU，可在脚本中手动设置 `device = torch.device('cpu')`。

---


这是使用PyTorch实现MSCRED

论文原文：
[http://in.arxiv.org/abs/1811.08055](http://in.arxiv.org/abs/1811.08055)

TensorFlow实现地址：
[https://github.com/7fantasysz/MSCRED](https://github.com/7fantasysz/MSCRED)

此项目就是通过上面tensorFlow转为Pytorch，具体流程如下：

- 先将时间序列数据转换为 image matrices

  > python ./utils/matrix_generator.py
  >
- 然后训练模型并对测试集生成相应的reconstructed matrices

  > python main.py
  >
- 最后评估模型，结果存在 `outputs`文件夹中

  > python ./utils/evaluate.py
  >
  > streamlit run visualization_app.py
  >
