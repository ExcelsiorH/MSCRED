import pandas as pd
import numpy as np
import os
import joblib
import glob
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm  # 如果没安装 tqdm，可以注释掉相关的进度条代码

# ================= 配置区域 =================

# 模式选择
# 'train': 读取训练目录所有文件 -> 聚合计算Scaler -> 保存Scaler -> 转换训练文件 -> 保存
# 'test':  读取Scaler -> 读取测试目录所有文件 -> 转换测试文件 -> 保存
MODE = 'test' 

# 路径配置
BASE_DIR = r'C:\Users\yusei\Workspace\异常检测\MSCRED\data'

# 输入目录 (请确保里面只有csv文件)
TRAIN_INPUT_DIR = os.path.join(BASE_DIR, 'train_raw')  # 存放原始训练CSV的文件夹
TEST_INPUT_DIR  = os.path.join(BASE_DIR, 'test_raw')   # 存放原始测试CSV的文件夹

# 输出目录
TRAIN_OUTPUT_DIR = os.path.join(BASE_DIR, 'train_processed')
TEST_OUTPUT_DIR  = os.path.join(BASE_DIR, 'test_processed')

# 模型权重保存路径
CHECKPOINT_DIR = './checkpoints'
SCALER_FILE = os.path.join(CHECKPOINT_DIR, 'stage_wise_scalers_multi.pkl')

# 排除列 (不进行归一化的列)
EXCLUDE_COLS = ['timestamp', 'stage'] 

# 归一化参数
FEATURE_RANGE = (-1, 1)

# ================= 核心逻辑 =================

def get_feature_columns(df):
    """自动识别需要归一化的特征列"""
    return [c for c in df.columns if c not in EXCLUDE_COLS]

def fit_global_scalers(file_paths):
    """
    第一遍扫描：聚合所有文件中相同 Stage 的数据，计算全局 Scaler
    """
    print(f"[Fit Phase] 开始扫描 {len(file_paths)} 个文件以计算全局统计量...")
    
    # 用于存储每个阶段的所有数据
    # 结构: { stage_id: [dataframe_1, dataframe_2, ...] }
    stage_buffer = {} 
    
    feature_cols = None

    for fp in tqdm(file_paths, desc="Loading files"):
        df = pd.read_csv(fp)
        
        if feature_cols is None:
            feature_cols = get_feature_columns(df)
            print(f"检测到特征列 ({len(feature_cols)}维): {feature_cols}")

        # 按 Stage 分组提取数据
        for stage_id, group_df in df.groupby('stage'):
            if stage_id not in stage_buffer:
                stage_buffer[stage_id] = []
            # 只取特征列存入 buffer
            stage_buffer[stage_id].append(group_df[feature_cols])

    # 开始计算 Scaler
    scalers_dict = {}
    print("\n[Fit Phase] 正在计算各阶段的 MinMax Scaler...")
    
    sorted_stages = sorted(stage_buffer.keys())
    for stage_id in sorted_stages:
        # 1. 拼接该阶段所有历史数据
        stage_all_data = pd.concat(stage_buffer[stage_id], axis=0)
        
        # 2. 初始化并训练 Scaler
        scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
        scaler.fit(stage_all_data)
        
        scalers_dict[stage_id] = scaler
        
        print(f"  Stage {stage_id}: 样本数={len(stage_all_data)}, "
              f"Max值示例={np.max(scaler.data_max_):.2f}, Min值示例={np.min(scaler.data_min_):.2f}")

    return scalers_dict, feature_cols

def process_and_save(file_paths, output_dir, scalers_dict, feature_cols):
    """
    第二遍扫描：使用计算好的 Scaler 对文件进行转换并保存
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\n[Transform Phase] 开始处理并保存文件到: {output_dir}")
    
    for fp in tqdm(file_paths, desc="Processing files"):
        df = pd.read_csv(fp)
        
        # 创建一个副本用于存储处理后的数据
        df_processed = df.copy()
        
        # 逐个 Stage 进行转换
        # 使用 .loc 确保数据回填到正确的位置，保持时间顺序不变
        for stage_id, scaler in scalers_dict.items():
            # 找到当前文件中属于该 stage 的行
            mask = df['stage'] == stage_id
            
            if not mask.any():
                continue
                
            # 提取原始数据
            # raw_values = df.loc[mask, feature_cols].values
            raw_df_slice = df.loc[mask, feature_cols] 
            # 归一化
            # transformed_values = scaler.transform(raw_values)
            transformed_values = scaler.transform(raw_df_slice)
            
            # (可选) 安全截断，防止测试集中出现极端异常值破坏数值稳定性
            # 训练好的范围内是 -1 到 1，这里放宽到 -10 到 10
            transformed_values = np.clip(transformed_values, -10, 10)
            
            # 回填数据
            df_processed.loc[mask, feature_cols] = transformed_values
            
        # 保存文件
        file_name = os.path.basename(fp)
        save_path = os.path.join(output_dir, f"processed_{file_name}")

        #去整个掉stage列再保存
        # df_processed = df_processed.drop(columns=['stage'])

        df_processed.to_csv(save_path, index=False)

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    if MODE == 'train':
        # 1. 获取所有训练文件
        train_files = glob.glob(os.path.join(TRAIN_INPUT_DIR, "*.csv"))
        if not train_files:
            raise ValueError(f"在 {TRAIN_INPUT_DIR} 没有找到csv文件")

        # 2. 计算 Scaler (Fit)
        scalers_dict, feature_cols = fit_global_scalers(train_files)
        
        # 3. 保存 Scaler
        joblib.dump({'scalers': scalers_dict, 'features': feature_cols}, SCALER_FILE)
        print(f"\nScaler 权重已保存至: {SCALER_FILE}")
        
        # 4. 转换训练文件 (Transform)
        process_and_save(train_files, TRAIN_OUTPUT_DIR, scalers_dict, feature_cols)

    elif MODE == 'test':
        # 1. 加载 Scaler
        if not os.path.exists(SCALER_FILE):
            raise FileNotFoundError(f"找不到 Scaler 文件: {SCALER_FILE}，请先运行 train 模式。")
        
        loaded_data = joblib.load(SCALER_FILE)
        scalers_dict = loaded_data['scalers']
        feature_cols = loaded_data['features']
        print(f"已加载 Scaler权重。特征维度: {len(feature_cols)}")
        
        # 2. 获取所有测试文件
        test_files = glob.glob(os.path.join(TEST_INPUT_DIR, "*.csv"))
        if not test_files:
            raise ValueError(f"在 {TEST_INPUT_DIR} 没有找到csv文件")
            
        # 3. 转换测试文件 (Transform)
        process_and_save(test_files, TEST_OUTPUT_DIR, scalers_dict, feature_cols)

if __name__ == "__main__":
    main()