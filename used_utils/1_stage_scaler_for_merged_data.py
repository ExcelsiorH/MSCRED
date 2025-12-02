import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# ================= 配置区域 =================

# 输入文件 (上一元生成的带 stage 的文件)
INPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1122\\0_merged_test_data1.csv'

# 输出目录和文件
CHECKPOINT_DIR = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\checkpoints'
OUTPUT_TRAIN_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1122\\0_merged_data_for_test1.csv'
SCALER_FILE = os.path.join(CHECKPOINT_DIR, 'stage_wise_scalers.pkl')

# 是否为测试文件 (如果为 True 则加载已保存的 scaler 并对数据做归一化；
# 如果为 False 则对每个 stage 拟合 scaler 并保存到 SCALER_FILE)
IS_TEST_FILE = True

# 所有的传感器特征列 (根据之前的脚本确定的列名)
# 脚本会自动从文件中排除 stage 和 timestamp，这里列出是为了明确特征范围
# 如果你的列名有变化，脚本逻辑通过 exclude 方式实现，无需硬编码列名

# ================= 核心处理逻辑 =================

def main():
    # 1. 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    # 2. 创建 checkpoints 目录
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"已创建目录: {CHECKPOINT_DIR}")

    print("正在读取数据...")
    df = pd.read_csv(INPUT_FILE)
    
    # 确保 timestamp 是 datetime 格式并设为索引
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

    # 识别特征列：排除 'stage' 列
    feature_cols = [col for col in df.columns if col != 'stage']
    print(f"检测到 {len(feature_cols)} 个特征传感器列: {feature_cols}")

    # 3. 分阶段归一化 (Stage-wise Normalization)
    print("\n开始执行分阶段归一化...")
    df_normalized_list = []

    # 如果是测试文件，先加载已保存的 scalers
    scalers_dict = {}
    if IS_TEST_FILE:
        if not os.path.exists(SCALER_FILE):
            print(f"错误: 测试模式下找不到 scaler 文件 {SCALER_FILE}，请先在训练模式下生成它。")
            return
        print(f"测试模式: 正在加载归一化权重 {SCALER_FILE} ...")
        scalers_dict = joblib.load(SCALER_FILE)

    # 按 stage 分组处理
    grouped = df.groupby('stage')

    for stage_id, group_df in grouped:
        stage_id = int(stage_id)
        print(f"  -> 处理阶段 {stage_id} (样本数: {len(group_df)})")

        # 提取特征数据
        X = group_df[feature_cols]

        if IS_TEST_FILE:
            # 测试时使用已保存的 scaler
            scaler = scalers_dict.get(stage_id)
            if scaler is None:
                print(f"  警告: 未找到阶段 {stage_id} 的已保存 scaler，回退为基于该阶段数据拟合新的 scaler。")
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = scaler.transform(X)
        else:
            # 训练/生成 scaler 并保存
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scalers_dict[stage_id] = scaler

        # 重构 DataFrame (保持索引不变)
        df_stage_scaled = pd.DataFrame(
            X_scaled,
            columns=feature_cols,
            index=group_df.index
        )

        # 把数值 stage 列加回去（保留为数值，而非独热编码）
        df_stage_scaled['stage'] = stage_id
        df_normalized_list.append(df_stage_scaled)

    # 合并所有阶段的数据并按时间重新排序
    df_final = pd.concat(df_normalized_list).sort_index()
    
    # 如果不是测试模式，保存 Scalers 到 checkpoints（测试模式下我们只是加载并使用它们）
    if not IS_TEST_FILE:
        joblib.dump(scalers_dict, SCALER_FILE)
        print(f"归一化权重已保存至: {SCALER_FILE}")

    # NOTE: 不进行独热编码，最终保留数值型 `stage` 列。
    df_encoded = df_final  # 名称兼容后续逻辑，实际为已归一化且保留 stage 列的数据框

    # 5. 最终检查与保存
    print("\n数据处理完成。")
    print(f"最终数据形状: {df_encoded.shape}")
    print(f"最终列名: {df_encoded.columns.tolist()}")
    
    # 检查是否有 NaN (归一化过程不应产生 NaN，除非某列方差为0)
    if df_encoded.isnull().values.any():
        print("警告: 最终数据中包含 NaN 值，请检查原始数据是否存在常数序列！")
        # 简单填充 (可选)
        # df_encoded = df_encoded.fillna(0)
    

    df_encoded.to_csv(OUTPUT_TRAIN_FILE)
    print(f"最终数据已保存至: {os.path.abspath(OUTPUT_TRAIN_FILE)}")


if __name__ == "__main__":
    main()