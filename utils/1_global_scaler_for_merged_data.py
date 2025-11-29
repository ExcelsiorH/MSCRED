import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# ================= 配置区域 =================

# 输入文件路径
INPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1122\\0_merged_test_data3.csv'

# 输出目录和文件
CHECKPOINT_DIR = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\checkpoints'
OUTPUT_TRAIN_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1122\\0_merged_data_for_test3_global.csv'
# 归一化权重保存路径 (建议重命名以区别于之前的 stage-wise scaler)
SCALER_FILE = os.path.join(CHECKPOINT_DIR, 'global_scaler.pkl')

# 是否为测试/验证文件 
# True: 加载已保存的全局 scaler 并对数据做归一化 (适用于测试集/验证集)
# False: 对当前数据拟合 scaler 并保存到 SCALER_FILE (适用于训练集)
IS_TEST_FILE = True

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

    # 识别特征列：排除 'stage' 列，只对传感器数值做归一化
    feature_cols = [col for col in df.columns if col != 'stage']
    print(f"检测到 {len(feature_cols)} 个特征传感器列")

    # 3. 全局归一化 (Global Normalization)
    print("\n开始执行全局归一化...")

    # 提取特征数据矩阵
    X = df[feature_cols]

    if IS_TEST_FILE:
        # ================= 测试/验证模式 =================
        if not os.path.exists(SCALER_FILE):
            print(f"错误: 测试模式下找不到 scaler 文件 {SCALER_FILE}，请先在训练模式下生成它。")
            return
        
        print(f"测试模式: 正在加载全局归一化权重 {SCALER_FILE} ...")
        scaler = joblib.load(SCALER_FILE)
        
        # 使用加载的 scaler 进行转换 (transform only)
        # 注意：如果列数不一致，这里会报错，需保证测试集特征与训练集一致
        X_scaled = scaler.transform(X)
        
    else:
        # ================= 训练模式 =================
        print("训练模式: 正在拟合新的全局 Scaler ...")
        scaler = StandardScaler()
        
        # 拟合并转换 (fit & transform)
        X_scaled = scaler.fit_transform(X)
        
        # 保存 Scaler 对象
        joblib.dump(scaler, SCALER_FILE)
        print(f"全局归一化权重已保存至: {SCALER_FILE}")

    # 4. 重构 DataFrame
    # 将归一化后的数据转回 DataFrame，保持索引不变
    df_encoded = pd.DataFrame(
        X_scaled,
        columns=feature_cols,
        index=df.index
    )

    # 把 'stage' 列加回去 (如果原始数据中有 stage)
    # 即使是全局归一化，通常也需要保留标签或阶段标识供后续模型使用
    if 'stage' in df.columns:
        df_encoded['stage'] = df['stage']

    # 5. 最终检查与保存
    print("\n数据处理完成。")
    print(f"最终数据形状: {df_encoded.shape}")
    
    # 检查是否有 NaN
    if df_encoded.isnull().values.any():
        print("警告: 最终数据中包含 NaN 值，请检查原始数据是否存在常数序列或极大异常值！")

    df_encoded.to_csv(OUTPUT_TRAIN_FILE)
    print(f"最终数据已保存至: {os.path.abspath(OUTPUT_TRAIN_FILE)}")


if __name__ == "__main__":
    main()