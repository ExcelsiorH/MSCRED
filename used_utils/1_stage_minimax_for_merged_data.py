import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

# ================= 配置区域 =================

# --- [核心开关] ---
IS_TEST_FILE = True  # 修改这里切换模式

if IS_TEST_FILE:
    INPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\0_merged_test_data_1128.csv'   # 你的测试集源文件
    OUTPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\0_test_data_stage_minimax_1128.csv'   # 输出的测试文件
else:
    INPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\0_merged_train_data_1121.csv' # 你的训练集源文件
    OUTPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\0_train_data_stage_minimax_1121.csv'   # 输出的训练文件

CHECKPOINT_DIR = './checkpoints'
SCALER_FILE = os.path.join(CHECKPOINT_DIR, 'stage_wise_scalers_robust.pkl')

# 训练集去噪参数
# 仅用于确定 scaler 的标尺，以及训练数据的清洗
CLIP_MIN_QUANTILE = 0.005  # 0.5% (稍微放宽一点，保留更多细节)
CLIP_MAX_QUANTILE = 0.995  # 99.5%
FEATURE_RANGE = (-1, 1)

# ================= 核心逻辑 =================

def main():
    mode_str = "【测试模式 TEST】" if IS_TEST_FILE else "【训练模式 TRAIN】"
    print(f"启动脚本: {mode_str}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    if not IS_TEST_FILE and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    if IS_TEST_FILE and not os.path.exists(SCALER_FILE):
        print(f"错误: 找不到权重文件 {SCALER_FILE}")
        return

    print(f"正在读取数据: {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
    
    feature_cols = [col for col in df.columns if col != 'stage']

    if IS_TEST_FILE:
        scalers_dict = joblib.load(SCALER_FILE)
    else:
        scalers_dict = {}

    print("\n开始分阶段处理...")
    df_processed_list = []
    
    if 'stage' not in df.columns:
        print("错误: 缺少 stage 列")
        return

    grouped = df.groupby('stage')
    
    for stage_id, group_df in grouped:
        stage_id = int(stage_id)
        X_stage = group_df[feature_cols].copy()
        
        # =========================================
        #         策略核心差异点
        # =========================================
        
        # --- 训练模式: 严格去噪 ---
        if not IS_TEST_FILE:
            # 1. 计算 99.5% 分位点，排除极端电气噪点
            lower_bound = X_stage.quantile(CLIP_MIN_QUANTILE)
            upper_bound = X_stage.quantile(CLIP_MAX_QUANTILE)
            
            # 2. 截断训练数据：保证输入模型的数据严格干净，分布在[-1, 1]
            X_stage_clean = X_stage.clip(lower=lower_bound, upper=upper_bound, axis=1)
            
            # 3. 拟合
            scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
            X_scaled_values = scaler.fit_transform(X_stage_clean)
            
            # 4. 保存 (Scaler 记住了 正常数据的 min 和 max)
            scalers_dict[stage_id] = {
                'scaler': scaler,
                'lower_bound': lower_bound, # 存着备用，但测试时不一定强制用
                'upper_bound': upper_bound
            }
            
        # --- 测试模式: 保留异常 ---
        else:
            if stage_id not in scalers_dict:
                continue
                
            params = scalers_dict[stage_id]
            scaler = params['scaler']
            # 注意：这里我不读取 bound 来做 clip 了
            
            # 1. 直接 Transform
            # 如果 X_stage 里有异常值(比如是正常值的2倍)，
            # transform 后它会变成 3.0 (或者是很大的数)，这正是我们想要的！
            X_scaled_values = scaler.transform(X_stage)
            
            # 【可选的安全措施】
            # 为了防止极其夸张的数值（如 NaN 或 1e9）导致程序崩溃，
            # 可以做一个极其宽泛的“安全截断”，比如限制在 [-10, 10]
            # 正常异常检测里，超过 [-1, 1] 很多就已经足够触发报警了。
            X_scaled_values = np.clip(X_scaled_values, -10, 10)

        # 重建 DataFrame
        df_stage_scaled = pd.DataFrame(
            X_scaled_values, 
            columns=feature_cols, 
            index=group_df.index
        )
        df_processed_list.append(df_stage_scaled)

    # 合并
    df_final = pd.concat(df_processed_list).sort_index()

    if not IS_TEST_FILE:
        joblib.dump(scalers_dict, SCALER_FILE)
        print(f"\n[训练模式] 权重已保存 (基于 0.5%-99.5% 去噪截断)")

    # 检查结果
    min_val = df_final.min().min()
    max_val = df_final.max().max()
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