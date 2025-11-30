import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
# 输入文件所在的文件夹路径
INPUT_DIR = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1122' 
# 输出文件名
OUTPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\0_merged_test_data3_1122.csv'

# CSV列名定义
TIME_COL = 'timestamp'      # 所有文件的时间戳列名
RAW_VALUE_COL = 'value'     # 12个基础传感器的数据列名

# 工艺工序时间表
PROCESS_SCHEDULE = [
    {"stage_id": 1, "start": "2025/11/22 17:24:10", "duration_min": 10, "desc": "工序1_低速正车"},
    {"stage_id": 2, "start": "2025/11/22 18:06:40", "duration_min": 15, "desc": "工序2_加盐2/3"},
    {"stage_id": 3, "start": "2025/11/22 18:26:35", "duration_min": 10, "desc": "工序3_加剩余盐"},
    {"stage_id": 4, "start": "2025/11/22 18:39:00", "duration_min": 15, "desc": "工序4_中速正车"},
    {"stage_id": 5, "start": "2025/11/22 18:56:00", "duration_min": 10, "desc": "工序5_低速正车"},
    {"stage_id": 6, "start": "2025/11/22 19:07:30", "duration_min": 15, "desc": "工序6_中速正车"},
    {"stage_id": 7, "start": "2025/11/22 19:23:30", "duration_min": 5,  "desc": "工序7_低速反车"},
    {"stage_id": 8, "start": "2025/11/22 19:29:10", "duration_min": 20, "desc": "工序8_中速正车"}
]
# ================= 传感器定义 =================

# 1. 12个基础传感器 (直接采样值，需降采样)
# 列表中的名字既是文件名(不带.csv)，也是合并后的目标列名
raw_sensors = [
    'flow_rate_inlet', 'flow_rate_outlet', 
    'pressure_inlet', 'pressure_outlet', 
    'rotation_speed', 
    'temp_barrel_bottom', 'temp_barrel_side', 'temp_inlet', 'temp_outlet', 
    'torque', 'vacuum','laser_distance'
]

# 2. 噪音传感器 (取 rms 列)
noise_sensor_file = 'noise_feature'
noise_col_name = 'rms'

# 3. 振动传感器 (取 mean 列并合成)
# 结构: {'新列名': [x文件名, y文件名, z文件名]}
vibration_groups = {
    'vibration1_mag': ['vibration1_x_feature', 'vibration1_y_feature', 'vibration1_z_feature'],
    'vibration2_mag': ['vibration2_x_feature', 'vibration2_y_feature', 'vibration2_z_feature']
}
vib_col_name = 'mean'

# ================= 核心处理函数 =================

def load_and_filter_schedule(filename, col_to_extract):
    """
    根据工序表提取数据，并确保索引排序，供 nearest 使用
    """
    path = os.path.join(INPUT_DIR, f"{filename}.csv")
    if not os.path.exists(path):
        print(f"[警告] 文件不存在，跳过: {path}")
        return None
    
    try:
        df = pd.read_csv(path, usecols=[TIME_COL, col_to_extract])
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        # nearest() 要求索引必须是排序的
        df = df.set_index(TIME_COL).sort_index() 
    except Exception as e:
        print(f"[错误] 读取 {filename} 失败: {e}")
        return None

    valid_segments = []
    
    for step in PROCESS_SCHEDULE:
        t_start = pd.to_datetime(step['start'])
        t_end = t_start + pd.Timedelta(minutes=step['duration_min'])
        
        # 为了配合 nearest，我们这里稍微放宽一点截取范围 (+/- 1秒)
        # 这样如果在边界上的点（比如 16:39:19.9）也能被取到
        mask_start = t_start - pd.Timedelta(seconds=1)
        mask_end = t_end + pd.Timedelta(seconds=1)
        
        segment = df[(df.index >= mask_start) & (df.index <= mask_end)]
        
        # 再次利用 mask 确保数据只落在我们宽泛的时间窗内
        # 具体的对齐交给后面的 resample
        if not segment.empty:
            valid_segments.append(segment)
    
    if not valid_segments:
        return None
    
    # 拼接
    df_combined = pd.concat(valid_segments)
    # 去重 (防止放宽时间窗导致的数据重叠)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    
    return df_combined

def process_raw_sensors():
    print(f"正在处理 12 个基础传感器 (使用 nearest 对齐)...")
    processed_data = []
    
    for sensor_name in raw_sensors:
        df = load_and_filter_schedule(sensor_name, col_to_extract=RAW_VALUE_COL)
        if df is not None and not df.empty:
            # === 关键修改 ===
            # 使用 nearest(limit=1) 寻找离整秒最近的点
            # limit=1 保证不会跨越太大的时间间隙去填充数据
            df_resampled = df.resample('1s').nearest(limit=1)
            
            # 过滤掉因为工序间隙产生的空行
            df_resampled = df_resampled.dropna()
            
            df_final = df_resampled.rename(columns={RAW_VALUE_COL: sensor_name})
            processed_data.append(df_final)
    return processed_data

def process_noise_sensor():
    print(f"正在处理噪音传感器...")
    df = load_and_filter_schedule(noise_sensor_file, col_to_extract=noise_col_name)
    if df is not None and not df.empty:
        # 即使是1Hz数据，用nearest也能修正 "19:00:00.8" 这种记录误差
        df = df.resample('1s').nearest(limit=1).dropna()
        df = df.rename(columns={noise_col_name: 'noise_rms'})
        return [df]
    return []

def process_vibration_sensors():
    print(f"正在处理振动传感器 (提取列: {vib_col_name} 并合成)...")
    processed_vibs = []
    
    for vib_target_name, files in vibration_groups.items():
        # files=[x_file, y_file, z_file]
        dfs = [load_and_filter_schedule(f, col_to_extract=vib_col_name) for f in files]
        
        # 检查是否三个文件都读取成功
        if all(d is not None and not d.empty for d in dfs):
            # 将时间戳按秒对齐：对每个分量按1s重采样（取该秒内的第一个样本）
            dfs_sec = [d.resample('1s').nearest(limit=1).dropna() for d in dfs]
            # 先取交集索引，确保xyz在同一时刻都有值才能计算
            common_idx = dfs_sec[0].index.intersection(dfs_sec[1].index).intersection(dfs_sec[2].index)
            
            # 提取对齐后的numpy数组
            x = dfs_sec[0].loc[common_idx][vib_col_name].values
            y = dfs_sec[1].loc[common_idx][vib_col_name].values
            z = dfs_sec[2].loc[common_idx][vib_col_name].values
            
            # 合成幅值
            mag = np.sqrt(x**2 + y**2 + z**2)
            
            # 构建新的Series
            df_mag = pd.DataFrame(mag, index=common_idx, columns=[vib_target_name])
            
            # 同样进行1s重采样对齐
            df_mag = df_mag.resample('1s').nearest(limit=1).dropna()
            
            processed_vibs.append(df_mag)
        else:
            print(f"[跳过] {vib_target_name} 某个分量文件缺失或为空")
            
    return processed_vibs

# ================= 主程序 =================

def main():
    all_series = []
    
    # 1. 执行处理
    all_series.extend(process_raw_sensors())
    all_series.extend(process_noise_sensor())
    all_series.extend(process_vibration_sensors())
    
    if not all_series:
        print("错误：未获取到任何有效数据。")
        return

    print(f"正在合并数据 (共 {len(all_series)} 列)...")
    
    # 2. 合并数据
    final_df = pd.concat(all_series, axis=1, join='inner')
    
    if final_df.empty:
        print("警告：合并后数据为空！")
        return

    # 3. 添加 Stage 列 (关键修改)
    print("正在添加 Stage 标签...")
    
    # 初始化一个 stage 列，默认值为 0 或 NaN
    final_df['stage'] = 0 
    
    # 创建一个用于保留行的 mask
    keep_mask = pd.Series(False, index=final_df.index)
    
    for step in PROCESS_SCHEDULE:
        t_start = pd.to_datetime(step['start'])
        t_end = t_start + pd.Timedelta(minutes=step['duration_min'])
        
        # 找到属于当前工序的时间索引
        # 这里的逻辑是：如果当前行的时间落在这个工序区间内，就标记上对应的 stage_id
        current_stage_mask = (final_df.index >= t_start) & (final_df.index <= t_end)
        
        # 赋值
        final_df.loc[current_stage_mask, 'stage'] = step['stage_id']
        
        # 更新总的保留 mask
        keep_mask |= current_stage_mask
            
    # 4. 过滤非工序时间的数据
    final_df = final_df[keep_mask]
    
    # 将 stage 列转为整数 (美观)
    final_df['stage'] = final_df['stage'].astype(int)

    print(f"合并成功！")
    print(f"总样本数: {len(final_df)}")
    print(f"数据列: {final_df.columns.tolist()}")
    
    # 简单统计一下每个阶段的点数，确认没切错
    print("各阶段数据量统计:")
    print(final_df['stage'].value_counts().sort_index())
    
    final_df.to_csv(OUTPUT_FILE)
    print(f"文件已保存至: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()