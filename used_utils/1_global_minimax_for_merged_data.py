import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

# ================= 配置区域 =================

# --- [核心开关] ---
IS_TEST_FILE = True  # 修改这里切换模式：False=训练(生成Scaler), True=测试(使用Scaler)

if IS_TEST_FILE:
    INPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1122\\0_merged_test_data3.csv'
    OUTPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1122\\0_merged_data_for_test3_global_minimax.csv'
else:
    INPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1121\\0_merged_data.csv'
    OUTPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1121\\0_merged_data_for_train_global_minimax.csv'

CHECKPOINT_DIR = './checkpoints'
# 修改文件名，避免与分阶段的权重文件混淆
SCALER_FILE = os.path.join(CHECKPOINT_DIR, 'global_scaler_minimax_robust.pkl')

# 训练集去噪参数 (逻辑保持不变，但应用范围变为全局)
CLIP_MIN_QUANTILE = 0.005  # 0.5%
CLIP_MAX_QUANTILE = 0.995  # 99.5%
FEATURE_RANGE = (-1, 1)

# ================= 核心逻辑 =================

def main():
    mode_str = "【测试模式 TEST】" if IS_TEST_FILE else "【训练模式 TRAIN】"
    print(f"启动脚本: {mode_str} (全局归一化 Global MinMax)")
    
    # 1. 基础检查
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    if not IS_TEST_FILE and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    if IS_TEST_FILE and not os.path.exists(SCALER_FILE):
        print(f"错误: 找不到全局权重文件 {SCALER_FILE}，请先运行训练模式生成它。")
        return

    # 2. 读取数据
    print(f"正在读取数据: {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
    
    # 识别特征列 (排除 stage)
    feature_cols = [col for col in df.columns if col != 'stage']
    print(f"检测到 {len(feature_cols)} 个特征列。")

    # 提取特征矩阵
    X = df[feature_cols].copy()

    # =========================================
    #         全局归一化核心逻辑
    # =========================================

    if not IS_TEST_FILE:
        # --- 训练模式: 全局去噪 & 拟合 ---
        print("正在计算全局统计量...")
        
        # 1. 计算全局分位点 (Global Quantiles)
        lower_bound = X.quantile(CLIP_MIN_QUANTILE)
        upper_bound = X.quantile(CLIP_MAX_QUANTILE)
        
        # 2. 全局截断 (Global Clip): 清洗极端噪点
        # 注意：这里是对整个数据集统一做 clip，不再区分 stage
        X_clean = X.clip(lower=lower_bound, upper=upper_bound, axis=1)
        
        # 3. 拟合全局 Scaler
        scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
        X_scaled_values = scaler.fit_transform(X_clean)
        
        # 4. 保存全局权重
        global_scaler_data = {
            'scaler': scaler,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        joblib.dump(global_scaler_data, SCALER_FILE)
        print(f"全局权重已保存至: {SCALER_FILE}")

    else:
        # --- 测试模式: 加载 & 转换 ---
        print("正在加载全局权重...")
        global_scaler_data = joblib.load(SCALER_FILE)
        scaler = global_scaler_data['scaler']
        
        # 1. 直接 Transform (保留异常值)
        # 不使用 saved bounds 进行 clip，允许测试数据中的异常值突破边界
        X_scaled_values = scaler.transform(X)
        
        # 2. 安全截断 (Safety Clip)
        # 防止数值爆炸导致 NaN 或 Inf，范围保持 [-10, 10]
        X_scaled_values = np.clip(X_scaled_values, -10, 10)

    # 3. 重建 DataFrame
    df_final = pd.DataFrame(
        X_scaled_values, 
        columns=feature_cols, 
        index=df.index
    )

    # 4. 把 'stage' 列补回去 (如果原数据有)
    # if 'stage' in df.columns:
    #     df_final['stage'] = df['stage']

    # 5. 检查与保存
    min_val = df_final[feature_cols].min().min()
    max_val = df_final[feature_cols].max().max()
    print(f"最终数据范围: Min={min_val:.4f}, Max={max_val:.4f}")
    
    if IS_TEST_FILE:
        if min_val < -1 or max_val > 1:
            print("提示: 测试数据存在超出 [-1, 1] 的值，这属于预期内的‘异常漂移’现象。")
    else:
        print("提示: 训练数据严格限制在 [-1, 1] 之间。")
    
    df_final.to_csv(OUTPUT_FILE)
    print(f"完成。文件已保存至: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()