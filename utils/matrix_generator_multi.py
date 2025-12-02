import numpy as np
import argparse
import pandas as pd
import os
import glob
from tqdm import tqdm

# ================= 参数配置 =================
parser = argparse.ArgumentParser(description='Signature Matrix Generator Modified for Multi-File')
parser.add_argument('--step_max', type=int, default=5,
                    help='ConvLSTM的时间步长 (Sequence Length)')
parser.add_argument('--gap_time', type=int, default=1, 
                    help='采样间隔 (Stride)，训练时建议设小一点以增加样本量')
parser.add_argument('--win_size', type=int, nargs='+', default=[10, 30, 60],
                    help='不同尺度的窗口大小 (Scale Windows)')
parser.add_argument('--raw_data_path', type=str, default='data/train_processed',
                    help='训练模式下填文件夹路径，测试模式下填具体文件路径')
parser.add_argument('--save_data_path', type=str, default='data/matrix_data',
                    help='保存数据的根目录')
parser.add_argument('--data_type', type=str, choices=['train', 'test'], default='train',
                    help='模式: train (多文件处理) 或 test (单文件处理)')

args = parser.parse_args()

# ================= 核心工具函数 =================

def generate_signature_matrix(data, win_sizes):
    """
    根据给定的窗口列表，生成多通道签名矩阵 (Signature Matrices)
    Input: data (Time_Window, Sensors)
    Output: (Channels, Sensors, Sensors)
    """
    sensors_count = data.shape[1]
    matrix_list = []
    
    for w in win_sizes:
        # 截取最后的 w 个时间步
        if w > len(data):
            # 如果历史数据不足 w，则取所有可用数据 (或根据需求补零，这里取所有可用)
            part_data = data
        else:
            part_data = data[-w:]
        
        # 计算特征间的相关性矩阵 (Transposed so sensors are rows for correlation)
        # np.corrcoef 计算的是行与行的相关性，所以我们要转置一下输入
        # rho = np.corrcoef(part_data.T)
        # 使用 errstate 上下文管理器忽略“除以0”和“无效值”的警告
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.corrcoef(part_data.T)
        
        # 处理可能出现的 NaN (如果某个传感器在这段时间内值完全不变，标准差为0，相关系数为NaN)
        rho = np.nan_to_num(rho, nan=0.0)
        
        matrix_list.append(rho)
    
    return np.array(matrix_list) # Shape: (Channel, N, N)

def process_single_file_logic(file_path, save_dir, win_size, step_max, gap_time, start_index=0):
    """
    处理单个文件的切片逻辑
    Returns:
        saved_count: 这个文件生成了多少个样本
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 自动排除非特征列
    exclude_cols = ['timestamp', 'stage']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    data_values = df[feature_cols].values
    
    num_samples = len(data_values)
    sensors_count = data_values.shape[1]
    
    # 计算最小需要的历史长度：最大的窗口 + (序列长度-1)*间隔
    # 比如 win=[10,30,60], step=5, gap=10
    # t=0时，我们需要往回找 step_max 个 steps。
    # 实际上 MSCRED 的输入是 shape (step_max, channels, H, W)
    # 我们需要保证即便是序列里的第一个时间步，也有足够的历史数据来计算最大的窗口。
    
    # max_win = win_size[-1]
    # min_start_idx = max_win + (step_max - 1) * gap_time 
    
    count = 0
    current_save_idx = start_index
    
    # 遍历文件中的时间点
    # 从这就开始切片，确保每个切片都有足够的数据
    # range(start, stop, step)
    
    # 为了方便，我们定义：如果当前时刻是 t，
    # 我们需要取 [t - (step_max-1)*gap, ..., t] 这 step_max 个时间点作为序列的锚点
    # 对每个锚点，再往前取 win_size 长度计算矩阵。
    
    # 因此，数据的结束点是 t，起始点至少要是: t - (step_max-1)*gap - max(win_size) >= 0
    max_w = win_size[-1]
    min_t = (step_max - 1) * gap_time + max_w
    
    # 如果文件太短，直接跳过
    if num_samples < min_t:
        print(f"  [Skip] 文件 {os.path.basename(file_path)} 太短，无法生成切片。")
        return 0

    print(f"  正在处理: {os.path.basename(file_path)} | 维度: {data_values.shape}")

    # 开始滑动
    for t in range(min_t, num_samples, gap_time):
        
        # 构建一个样本 sequence
        # Shape: (step_max, channels, sensors, sensors)
        sequence_matrices = []
        
        # 回溯生成 step_max 个时间步的数据
        for s in range(step_max):
            # 对应的每个时间步的"当前时刻"
            # 注意顺序：我们要生成 [t-(step-1), ..., t] 还是 [t, t-1, ...]
            # ConvLSTM 通常输入是时序正向的。
            # 假设 t 是当前时刻（序列最后一步），则序列第一步是 t - (step_max - 1 - s) * gap_time
            
            curr_anchor_idx = t - (step_max - 1 - s) * gap_time
            
            # 截取历史数据用于计算矩阵：从 curr_anchor_idx 往前取 max_w 长度
            # 实际上 generate_signature_matrix 内部会处理不同窗口
            # 我们只需要把截止到 curr_anchor_idx 的数据传进去即可
            # 注意：切片是 左闭右开，所以用 :curr_anchor_idx+1
            history_data = data_values[:curr_anchor_idx+1]
            
            # 生成该时刻的多通道矩阵
            matrix = generate_signature_matrix(history_data, win_size)
            sequence_matrices.append(matrix)
            
        # 转换为 array
        # Shape: (step_max, Channels, N, N)
        sequence_tensor = np.array(sequence_matrices)
        
        # 保存
        # 训练集用全局索引 (0.npy, 1.npy...)
        # 测试集通常不需要这么做，但为了统一，这里只在文件名上做区分
        
        save_name = os.path.join(save_dir, f"{current_save_idx}.npy")
        np.save(save_name, sequence_tensor)
        
        current_save_idx += 1
        count += 1
        
    return count

# ================= 主流程 =================

def main():
    print(f"Mode: {args.data_type}")
    print(f"Raw Data Path: {args.raw_data_path}")
    
    win_size = args.win_size
    step_max = args.step_max
    gap_time = args.gap_time
    
    if args.data_type == 'train':
        # 1. 准备目录
        # 训练集输出目录: data/train_data/
        train_output_dir = os.path.join(args.save_data_path, 'train_data')
        if not os.path.exists(train_output_dir):
            os.makedirs(train_output_dir)
            
        # 2. 获取所有 CSV 文件
        file_list = sorted(glob.glob(os.path.join(args.raw_data_path, "*.csv")))
        if not file_list:
            raise ValueError(f"未在 {args.raw_data_path} 找到CSV文件")
            
        print(f"发现 {len(file_list)} 个训练文件。开始构建训练集...")
        
        global_idx = 0
        
        # 3. 循环处理
        for fp in tqdm(file_list):
            generated_count = process_single_file_logic(
                fp, train_output_dir, win_size, step_max, gap_time, start_index=global_idx
            )
            global_idx += generated_count
            
        print(f"\n[完成] 训练集构建完毕。")
        print(f"共生成样本数: {global_idx}")
        print(f"保存路径: {train_output_dir}")
        print(f"样本形状示例: (sequence_len={step_max}, channels={len(win_size)}, N, N)")
        
    else:
        # ================= 测试模式 =================
        # 测试模式保持单文件逻辑，通常不需要把所有测试文件拼成一个大文件夹
        # 而是针对这一次实验生成一组数据
        
        if not os.path.isfile(args.raw_data_path):
             raise ValueError("测试模式下，raw_data_path 必须是一个具体的文件路径")
             
        file_name = os.path.basename(args.raw_data_path).split('.')[0]
        test_output_dir = os.path.join(args.save_data_path, f'test_{file_name}')
        
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
            
        print(f"正在处理单个测试文件: {args.raw_data_path}")
        
        # 对于测试集，索引从 0 开始即可
        count = process_single_file_logic(
            args.raw_data_path, test_output_dir, win_size, step_max, gap_time, start_index=0
        )
        
        print(f"\n[完成] 测试集构建完毕。样本数: {count}")
        print(f"保存路径: {test_output_dir}")

if __name__ == "__main__":
    main()