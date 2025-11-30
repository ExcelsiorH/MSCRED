import numpy as np
import argparse
import pandas as pd
import os
import math

parser = argparse.ArgumentParser(description='Signature Matrix Generator Modified')
parser.add_argument('--ts_type', type=str, default="node",
                    help='type of time series: node or link')
parser.add_argument('--step_max', type=int, default=5,
                    help='maximum step in ConvLSTM')
parser.add_argument('--gap_time', type=int, default=1,  # stride width
                    help='gap time between each segment')
parser.add_argument('--win_size', type=int, nargs='+', default=[10, 30, 60],
                    help='window size of each segment')
parser.add_argument('--raw_data_path', type=str, default='data/origin_merged_data/0_test_data3_stage_minimax_1122.csv',
                    help='path to load raw data (csv file)') #required=True,
parser.add_argument('--save_data_path', type=str, default='data/',
                    help='path to save data')
# 新增参数：指定当前处理的数据类型（训练集或测试集）
parser.add_argument('--data_type', type=str, choices=['train', 'test'],default='test',
                    help='process type: train or test') #required=True,

args = parser.parse_args()
print("Arguments:", args)

ts_type = args.ts_type
step_max = args.step_max
gap_time = args.gap_time
win_size = args.win_size
raw_data_path = args.raw_data_path
save_data_path = args.save_data_path
data_type = args.data_type

# 初始化路径
matrix_data_path = os.path.join(save_data_path, "matrix_data")
if not os.path.exists(matrix_data_path):
    os.makedirs(matrix_data_path)

# 全局变量，用于在两个函数间传递实际的数据长度
global_data_len = 0

def generate_signature_matrix_node():
    global global_data_len
    print(f"Loading data from {raw_data_path}...")
    
    # 修改1：读取带表头的CSV，并处理格式
    # header=0 表示第一行是表头
    df = pd.read_csv(raw_data_path, header=0)
    
    # 修改2：去除第一列时间戳，保留后15列传感器数据
    # 假设格式：[Timestamp, Sensor1, Sensor2, ..., Sensor15]
    data_values = df.iloc[:, 1:].values
    
    # 修改3：转置数据
    # 原始脚本逻辑期望数据形状为 (sensor_n, time_length)
    # 而 pandas读取后通常是 (time_length, sensor_n)，所以需要转置
    data = np.transpose(data_values).astype(np.float64)
    
    sensor_n = data.shape[0]
    time_len = data.shape[1]
    global_data_len = time_len # 记录实际时间长度
    
    print(f"Data Shape (Sensor x Time): {data.shape}")

    # 修改4：已做过归一化，此处移除 Min-Max Normalization 代码
    # data = (np.transpose(data) - min_value)/(max_value - min_value + 1e-6) ... (Removed)

    # 动态设定 min_time 和 max_time
    min_time = 0
    max_time = time_len

    # multi-scale signature matrix generation (核心逻辑保持不变)
    for w in range(len(win_size)):
        matrix_all = []
        win = win_size[w]
        print("generating signature with window " + str(win) + "...")
        
        # 按照 gap_time 步长遍历时间
        for t in range(min_time, max_time, gap_time):
            matrix_t = np.zeros((sensor_n, sensor_n))
            
            # 只有当时间点大于等于窗口大小时才能计算相关性
            if t >= win:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        # Inner product calculation
                        matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t]) / win
                        matrix_t[j][i] = matrix_t[i][j]
            
            matrix_all.append(matrix_t)
        
        # 保存中间矩阵文件
        # 注意：这里会覆盖之前生成的同名文件，所以处理完 train 必须处理完后续步骤再处理 test
        # 或者可以修改文件名加上 data_type 前缀，但为了保持后续函数不改动太大，暂保持原样
        path_temp = os.path.join(matrix_data_path, "matrix_win_" + str(win))
        np.save(path_temp, matrix_all)
        del matrix_all[:]

    print("matrix generation finish!")

def generate_train_test_data():
    # data sample generation
    print(f"generating {data_type} data samples sequences...")
    
    # 根据 data_type 决定保存路径
    output_dir = os.path.join(matrix_data_path, f"{data_type}_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_all = []
    # 加载上一步生成的中间矩阵
    for w in range(len(win_size)):
        path_temp = os.path.join(matrix_data_path, "matrix_win_" + str(win_size[w]) + ".npy")
        if not os.path.exists(path_temp):
             print(f"Error: Intermediate file {path_temp} not found. Run matrix generation first.")
             return
        data_all.append(np.load(path_temp))

    # 计算总的时间步数 (chunks)
    # 因为 generate_signature_matrix_node 是按 gap_time 采样的，所以这里列表长度就是 time_len / gap_time
    total_chunks = len(data_all[0])
    
    # 遍历所有生成的矩阵块
    for data_id in range(total_chunks):
        step_multi_matrix = []
        
        # 构建时序序列 (Sequence Generation for ConvLSTM)
        # 需回溯 step_max 个时间步
        for step_id in range(step_max, 0, -1):
            multi_matrix = []
            for i in range(len(win_size)):
                # 检查索引是否越界
                idx = data_id - step_id
                if idx < 0:
                    # 如果历史数据不足（刚开始的几个点），补零矩阵或跳过
                    # 原逻辑未显式处理 padding，通常会依靠下面的 valid check 跳过
                    # 这里为了防止 crash，简单处理为全0矩阵 (或取第0个)
                    # 按照原代码逻辑，它是在下面 if 判断里过滤掉无效数据的
                    if idx < 0: idx = 0 
                    multi_matrix.append(data_all[i][idx])
                else:
                    multi_matrix.append(data_all[i][idx])
            step_multi_matrix.append(multi_matrix)

        # 核心逻辑：判断当前 data_id 是否有效
        # 有效条件：当前索引必须足以覆盖最大的窗口 + 序列长度
        # global_data_len 是原始数据点数， data_id 是 gap_time 步长后的索引
        # 还原回原始时间点: current_raw_time = data_id * gap_time
        
        min_valid_raw_time = win_size[-1] + step_max * gap_time
        current_raw_time = data_id * gap_time

        if current_raw_time >= min_valid_raw_time:
            # 保存文件，文件名为该文件内的相对索引
            save_name = f"{data_type}_data_{data_id}"
            path_temp = os.path.join(output_dir, save_name)
            np.save(path_temp, step_multi_matrix)
        
        del step_multi_matrix[:]

    print(f"{data_type} data generation finish! Saved to {output_dir}")

if __name__ == '__main__':
    if ts_type == "node":
        generate_signature_matrix_node()

    generate_train_test_data()