import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# ================= 配置区域 =================

# 输入文件 (上一元生成的带 stage 的文件)
INPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1121\\0_merged_data_with_stage.csv'

# 输出目录和文件
CHECKPOINT_DIR = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\checkpoints'
OUTPUT_TRAIN_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1121\\0_merged_data_for_train.csv'
SCALER_FILE = os.path.join(CHECKPOINT_DIR, 'stage_wise_scalers.pkl')

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
    scalers_dict = {} # 用于保存每个阶段的 scaler

    # 按 stage 分组处理
    # 假设 stage 是 1 到 8
    grouped = df.groupby('stage')
    
    for stage_id, group_df in grouped:
        stage_id = int(stage_id)
        print(f"  -> 处理阶段 {stage_id} (样本数: {len(group_df)})")
        
        # 提取特征数据
        X = group_df[feature_cols]
        
        # 初始化并拟合该阶段的 Scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 保存 scaler (重要：推理时需要用到)
        scalers_dict[stage_id] = scaler
        
        # 重构 DataFrame (保持索引不变)
        df_stage_scaled = pd.DataFrame(
            X_scaled, 
            columns=feature_cols, 
            index=group_df.index
        )
        
        # 把 stage 列加回去，用于下一步编码
        df_stage_scaled['stage'] = stage_id
        
        df_normalized_list.append(df_stage_scaled)

    # 合并所有阶段的数据并按时间重新排序
    df_final = pd.concat(df_normalized_list).sort_index()
    
    # 保存 Scalers 到 checkpoints
    joblib.dump(scalers_dict, SCALER_FILE)
    print(f"归一化权重已保存至: {SCALER_FILE}")

    # 4. Stage 编码 (One-Hot Encoding)
    print("\n正在对 Stage 列进行独热编码 (One-Hot Encoding)...")
    
    # 使用 get_dummies 进行编码，前缀设为 stage
    # columns=['stage'] 会自动移除原始 'stage' 列并替换为 stage_1, stage_2 等
    df_encoded = pd.get_dummies(df_final, columns=['stage'], prefix='stage')
    
    # 强制转换 One-Hot 列为 int (0/1) 而不是 True/False，兼容性更好
    stage_columns = [col for col in df_encoded.columns if col.startswith('stage_')]
    df_encoded[stage_columns] = df_encoded[stage_columns].astype(int)

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
    print(f"训练数据已保存至: {os.path.abspath(OUTPUT_TRAIN_FILE)}")
    print("-" * 30)
    print("现在你可以直接将此文件输入模型进行训练了。")

if __name__ == "__main__":
    main()