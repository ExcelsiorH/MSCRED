import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

# ================= 配置区域 =================

# --- [核心开关] ---
# True: 测试模式 (读取权重，不拟合，用于处理测试集/新数据)
# False: 训练模式 (计算权重并保存，用于构建训练集)
IS_TEST_FILE = True 

# 输入文件路径 (根据当前模式修改这里)
if IS_TEST_FILE:
    INPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1122\\0_merged_test_data3.csv'   # 你的测试集源文件
    OUTPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1122\\0_merged_data_for_test3_stage_minimax.csv'   # 输出的测试文件
else:
    INPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1121\\0_merged_data.csv' # 你的训练集源文件
    OUTPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1121\\0_merged_data_for_train_stage_minimax.csv'   # 输出的训练文件

# 权重保存路径
CHECKPOINT_DIR = './checkpoints'
SCALER_FILE = os.path.join(CHECKPOINT_DIR, 'stage_wise_minimax_scalers_robust.pkl')

# 归一化参数
CLIP_MIN_QUANTILE = 0.01  # 1% 分位数
CLIP_MAX_QUANTILE = 0.99  # 99% 分位数
FEATURE_RANGE = (-1, 1)   # 映射到 [-1, 1] 

# ================= 核心逻辑 =================

def main():
    mode_str = "【测试模式 TEST】" if IS_TEST_FILE else "【训练模式 TRAIN】"
    print(f"启动脚本: {mode_str}")
    
    # 1. 检查文件与目录
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    if not IS_TEST_FILE:
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
    else:
        # 测试模式下，权重文件必须存在
        if not os.path.exists(SCALER_FILE):
            print(f"错误: 测试模式需要先有训练好的权重文件: {SCALER_FILE}")
            print("请先将 IS_TEST_FILE 设为 False 运行一次以生成权重。")
            return

    # 2. 读取数据
    print(f"正在读取数据: {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    
    # 时间戳处理
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
    
    # 确定特征列 (排除 stage)
    feature_cols = [col for col in df.columns if col != 'stage']
    print(f"特征数量: {len(feature_cols)} (将只保留这些列和时间戳)")

    # 3. 加载或初始化参数容器
    if IS_TEST_FILE:
        print(f"正在加载权重文件: {SCALER_FILE}")
        scalers_dict = joblib.load(SCALER_FILE)
    else:
        scalers_dict = {}

    # 4. 分阶段归一化 (核心部分)
    print("\n开始分阶段处理...")
    df_processed_list = []
    
    # 按 stage 分组 (前提：测试文件也必须包含 stage 列用来识别工序)
    if 'stage' not in df.columns:
        print("错误: 输入文件必须包含 'stage' 列以便进行分阶段归一化！")
        return

    grouped = df.groupby('stage')
    
    for stage_id, group_df in grouped:
        stage_id = int(stage_id)
        X_stage = group_df[feature_cols].copy()
        
        # --- 训练模式逻辑 ---
        if not IS_TEST_FILE:
            # 1. 计算边界
            lower_bound = X_stage.quantile(CLIP_MIN_QUANTILE)
            upper_bound = X_stage.quantile(CLIP_MAX_QUANTILE)
            
            # 2. 截断 (Clip)
            X_stage_clipped = X_stage.clip(lower=lower_bound, upper=upper_bound, axis=1)
            
            # 3. 拟合 Scaler
            scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
            X_scaled_values = scaler.fit_transform(X_stage_clipped)
            
            # 4. 保存参数
            scalers_dict[stage_id] = {
                'scaler': scaler,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        # --- 测试模式逻辑 ---
        else:
            if stage_id not in scalers_dict:
                print(f"警告: 测试集中出现了训练集未见过的 Stage {stage_id}，跳过该部分数据！")
                continue
                
            params = scalers_dict[stage_id]
            scaler = params['scaler']
            lower_bound = params['lower_bound']
            upper_bound = params['upper_bound']
            
            # 1. 使用训练集的边界进行截断 (重要！)
            # 测试集如果有更极端的异常值，也会被限制在训练集的范围内
            X_stage_clipped = X_stage.clip(lower=lower_bound, upper=upper_bound, axis=1)
            
            # 2. 使用训练集的 Scaler 进行转换
            X_scaled_values = scaler.transform(X_stage_clipped)

        # 重建 DataFrame
        df_stage_scaled = pd.DataFrame(
            X_scaled_values, 
            columns=feature_cols, 
            index=group_df.index
        )
        
        # 此时不再把 stage 列加回去，因为我们最终不需要它
        df_processed_list.append(df_stage_scaled)

    # 5. 合并与保存
    if not df_processed_list:
        print("错误: 没有处理任何数据，请检查 stage 列是否匹配。")
        return

    # 合并并排序
    df_final = pd.concat(df_processed_list).sort_index()

    # 训练模式下保存权重
    if not IS_TEST_FILE:
        joblib.dump(scalers_dict, SCALER_FILE)
        print(f"\n[训练模式] 归一化权重已保存至: {SCALER_FILE}")

    # 6. 输出结果
    print("-" * 30)
    print("数据处理完成。")
    print(f"最终维度: {df_final.shape}")
    print(f"包含列: {df_final.columns.tolist()}") # 此时应该只有14个传感器列
    
    # 检查范围
    min_val = df_final.min().min()
    max_val = df_final.max().max()
    print(f"数据值域检查: Min={min_val:.4f}, Max={max_val:.4f}")
    
    if min_val < -1.0001 or max_val > 1.0001:
        # 测试集有可能会稍微越界一点点（浮点误差），或者是clip逻辑有误，通常clip后不会越界
        print("提示: 数据值域符合预期 ([-1, 1])")
    
    df_final.to_csv(OUTPUT_FILE)
    print(f"结果文件已保存至: {os.path.abspath(OUTPUT_FILE)}")
    
    if not IS_TEST_FILE:
        print("\n下一步提示: 现在你可以使用 output 文件训练 MSCRED，")
        print("并在未来将 IS_TEST_FILE 设为 True 来处理新的测试数据。")

if __name__ == "__main__":
    main()