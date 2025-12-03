import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
# 输入文件所在的文件夹路径
INPUT_DIR = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\origin_data\\1129' 
# 输出文件名
OUTPUT_FILE = 'C:\\Users\\yusei\\Workspace\\异常检测\\MSCRED\\data\\test_raw\\merged_test_data_1129_gearbox.csv'


# 工艺工序时间表
PROCESS_SCHEDULE = [
    {"stage_id": 1, "start": "2025/11/29 19:17:44", "duration_min": 10, "desc": "工序1_低速正车"},
    {"stage_id": 2, "start": "2025/11/29 19:47:50", "duration_min": 15, "desc": "工序2_加盐2/3"},
    {"stage_id": 3, "start": "2025/11/29 20:07:00", "duration_min": 10, "desc": "工序3_加剩余盐"},
    {"stage_id": 4, "start": "2025/11/29 20:19:40", "duration_min": 15, "desc": "工序4_中速正车"},
    {"stage_id": 5, "start": "2025/11/29 20:35:30", "duration_min": 10, "desc": "工序5_低速正车"},
    {"stage_id": 6, "start": "2025/11/29 20:47:00", "duration_min": 15, "desc": "工序6_中速正车"},
    {"stage_id": 7, "start": "2025/11/29 21:05:00", "duration_min": 5,  "desc": "工序7_低速反车"},
    {"stage_id": 8, "start": "2025/11/29 21:11:00", "duration_min": 20, "desc": "工序8_中速正车"}
]
# ================= 传感器定义 =================

# 1. 12个基础传感器 (直接采样值，需降采样)
# 列表中的名字既是文件名(不带.csv)，也是合并后的目标列名
raw_sensors = [
    'flow_rate_inlet', 'flow_rate_outlet', 
    'pressure_inlet', 'pressure_outlet', 
    'rotation_speed', 
    'temp_barrel_bottom', 'temp_barrel_side', 'temp_inlet', 'temp_outlet', 
    'torque', 'vacuum','laser_spacing'
]
# CSV列名定义
TIME_COL = 'timestamp'      # 所有文件的时间戳列名
RAW_VALUE_COL = 'value'     # 12个基础传感器的数据列名

# 2. 噪音传感器 (取 rms 列)
noise_sensor_file = 'noise_feature'
noise_col_name = 'rms'

# 3. [修改] 振动传感器 (取 rms_freq 列，不再合并，直接独立使用)
# 这里列出所有需要单独作为特征列的振动文件
vibration_files = [
    'vibration1_x_feature', 'vibration1_y_feature', 'vibration1_z_feature',
    'vibration2_x_feature', 'vibration2_y_feature', 'vibration2_z_feature'
]
# [修改] 提取的列名改为 rms_freq
vib_col_name = 'rms_freq'

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
        mask_start = t_start - pd.Timedelta(seconds=1)
        mask_end = t_end + pd.Timedelta(seconds=1)
        
        segment = df[(df.index >= mask_start) & (df.index <= mask_end)]
        
        if not segment.empty:
            valid_segments.append(segment)
    
    if not valid_segments:
        return None
    
    # 拼接
    df_combined = pd.concat(valid_segments)
    # 去重
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    
    return df_combined

def process_raw_sensors():
    print(f"正在处理 12 个基础传感器 (使用 nearest 对齐)...")
    processed_data = []
    
    for sensor_name in raw_sensors:
        df = load_and_filter_schedule(sensor_name, col_to_extract=RAW_VALUE_COL)
        if df is not None and not df.empty:
            # 使用 nearest(limit=1) 寻找离整秒最近的点
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
        df = df.resample('1s').nearest(limit=1).dropna()
        df = df.rename(columns={noise_col_name: 'noise_rms'})
        return [df]
    return []

def process_vibration_sensors():
    """
    [修改] 不再进行 XYZ 合成，而是直接提取每个文件的 rms_freq 并对齐
    """
    print(f"正在处理振动传感器 (独立提取列: {vib_col_name})...")
    processed_vibs = []
    
    for vib_file in vibration_files:
        # 读取指定列 (rms_freq)
        df = load_and_filter_schedule(vib_file, col_to_extract=vib_col_name)
        
        if df is not None and not df.empty:
            # 同样进行 1s 对齐
            df_resampled = df.resample('1s').nearest(limit=1).dropna()
            
            # 将列名重命名为文件名（例如 vibration1_x_feature），以便在合并时区分
            df_final = df_resampled.rename(columns={vib_col_name: vib_file})
            processed_vibs.append(df_final)
        else:
            print(f"[跳过] 振动文件缺失或为空: {vib_file}")
            
    return processed_vibs

# ================= 主程序 =================

def main():
    all_series = []
    
    # 1. 执行处理
    all_series.extend(process_raw_sensors())
    all_series.extend(process_noise_sensor())
    all_series.extend(process_vibration_sensors()) # 这里调用的已经是修改后的函数
    
    if not all_series:
        print("错误：未获取到任何有效数据。")
        return

    print(f"正在合并数据 (共 {len(all_series)} 列)...")
    
    # 2. 合并数据 (inner join 确保所有传感器在这一秒都有值)
    final_df = pd.concat(all_series, axis=1, join='inner')
    
    if final_df.empty:
        print("警告：合并后数据为空！可能原因是各传感器时间戳无法对齐（交集为空）。")
        return

    # 3. 添加 Stage 列
    print("正在添加 Stage 标签...")
    
    final_df['stage'] = 0 
    keep_mask = pd.Series(False, index=final_df.index)
    
    for step in PROCESS_SCHEDULE:
        t_start = pd.to_datetime(step['start'])
        t_end = t_start + pd.Timedelta(minutes=step['duration_min'])
        
        current_stage_mask = (final_df.index >= t_start) & (final_df.index <= t_end)
        final_df.loc[current_stage_mask, 'stage'] = step['stage_id']
        keep_mask |= current_stage_mask
            
    # 4. 过滤非工序时间的数据
    final_df = final_df[keep_mask]
    final_df['stage'] = final_df['stage'].astype(int)

    print(f"合并成功！")
    print(f"总样本数: {len(final_df)}")
    print(f"数据列: {final_df.columns.tolist()}")
    
    print("各阶段数据量统计:")
    print(final_df['stage'].value_counts().sort_index())
    
    final_df.to_csv(OUTPUT_FILE)
    print(f"文件已保存至: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()