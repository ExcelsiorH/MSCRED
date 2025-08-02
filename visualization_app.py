import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd

def load_anomaly_scores(matrix_data_path, valid_start, valid_end, test_start, test_end, gap_time, thred_b, duration_id):
    valid_anomaly_score = np.zeros((valid_end - valid_start, 1))
    test_anomaly_score = np.zeros((test_end - test_start, 1))
    
    test_data_path = matrix_data_path + "test_data//"
    reconstructed_data_path = matrix_data_path + "reconstructed_data//"
    matrix_test_mse=[]
    for i in range(valid_start, test_end):
        path_temp_1 = os.path.join(test_data_path, "test_data_" + str(i) + '.npy')
        gt_matrix_temp = np.load(path_temp_1)
        
        path_temp_2 = os.path.join(reconstructed_data_path, "reconstructed_data_" + str(i) + '.npy')
        reconstructed_matrix_temp = np.load(path_temp_2)
        
        select_gt_matrix = np.array(gt_matrix_temp)[-1][duration_id]
        select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][duration_id]
        
        select_matrix_error = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix))
        num_broken = len(select_matrix_error[select_matrix_error > thred_b])
        if i>=valid_end:
            matrix_test_mse.append(select_matrix_error)
        if i < valid_end:
            valid_anomaly_score[i - valid_start] = num_broken
        else:
            test_anomaly_score[i - test_start] = num_broken
            
    return valid_anomaly_score, test_anomaly_score, matrix_test_mse

def load_anomaly_ground_truth(file_path, test_start, gap_time):
    anomaly_pos = np.zeros(5)
    root_cause_gt = np.zeros((5, 3))
    anomaly_span = [10, 30, 90]
    
    with open(file_path, "r") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            anomaly_axis = int(line.split(',')[0])
            anomaly_pos[row_index] = anomaly_axis/gap_time - test_start - anomaly_span[row_index%3]/gap_time
            
            root_list = line.split(',')[1:]
            for k in range(len(root_list)-1):
                root_cause_gt[row_index][k] = int(root_list[k])
                
    return anomaly_pos, root_cause_gt, anomaly_span

def plot_anomaly_scores(test_anomaly_score, valid_anomaly_max, alpha, anomaly_pos, anomaly_span, gap_time):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制异常分数曲线
    ax.plot(test_anomaly_score, color='black', linewidth=2, label='Anomaly Score')
    
    # 绘制阈值线
    threshold = np.full((len(test_anomaly_score)), valid_anomaly_max * alpha)
    ax.plot(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    
    # 标记异常时间段
    for k in range(len(anomaly_pos)):
        ax.axvspan(anomaly_pos[k], anomaly_pos[k] + anomaly_span[k%3]/gap_time, 
                  color='red', alpha=0.3, label='Anomaly Period' if k==0 else "")
    
    ax.set_xlabel('Test Time', fontsize=12)
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    
    return fig

def plot_error_heatmap(error_matrix, timestep_index):
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(error_matrix, cmap='jet')
    fig.colorbar(cax, ax=ax)
    ax.set_title(f'Error Heatmap at t={timestep_index}')
    #root cause analysis
    error_threshold_ratio = 0.6  # 设置误差阈值比例
    row_sum = np.sum(error_matrix, axis=1)
    col_sum = np.sum(error_matrix, axis=0)

    row_threshold = error_threshold_ratio * np.max(row_sum)
    col_threshold = error_threshold_ratio * np.max(col_sum)

    root_cause_row_indices = np.where(row_sum >= row_threshold)[0]
    root_cause_col_indices = np.where(col_sum >= col_threshold)[0]
    if len(root_cause_row_indices)> len(root_cause_col_indices):
        root_cause_indices= root_cause_row_indices
    else:
        root_cause_indices= root_cause_col_indices
    return fig,root_cause_indices

def main():
    st.title("MSCRED异常检测可视化界面")
    
    # 侧边栏参数设置
    st.sidebar.header("参数设置")
    thred_b = st.sidebar.slider(
        "异常阈值 (thred_broken)", 
        min_value=0.001, 
        max_value=0.01, 
        value=0.005, 
        step=0.001, 
        format="%.3f"
    )
    alpha = st.sidebar.slider("阈值系数 (alpha)", 1.0, 2.0, 1.5, 0.1)
    gap_time = 10
    #设置异常持续类型S，M，L对应0，1，2
    anomaly_duration = st.sidebar.selectbox("异常持续类型(anomaly duration)", ["Short", "Medium", "Long"], index=0)
    if anomaly_duration == "Short":
        duration_id=0
    elif anomaly_duration == "Medium":
        duration_id=1
    else:
        duration_id=2
    # 数据加载
    matrix_data_path = 'data//matrix_data//'
    valid_start = 8000 // gap_time
    valid_end = 10000 // gap_time
    test_start = 10000 // gap_time
    test_end = 20000 // gap_time
    
    try:
        valid_anomaly_score, test_anomaly_score, matrix_test_mse = load_anomaly_scores(
            matrix_data_path, valid_start, valid_end, test_start, test_end, gap_time, thred_b,duration_id
        )
        
        valid_anomaly_max = np.max(valid_anomaly_score.ravel())
        test_anomaly_score = test_anomaly_score.ravel()
        
        anomaly_pos, root_cause_gt, anomaly_span = load_anomaly_ground_truth(
            "data//test_anomaly.csv", test_start, gap_time
        )
        
        # 绘制图表
        fig = plot_anomaly_scores(
            test_anomaly_score, valid_anomaly_max, alpha, anomaly_pos, anomaly_span, gap_time
        )
        st.pyplot(fig)
        
        anomaly_collection=[]
        for i in range(len(matrix_test_mse)):
            if test_anomaly_score[i]>valid_anomaly_max * alpha:
                # plot_error_heatmap(matrix_test_mse[i], i + test_start)
                anomaly_collection.append((matrix_test_mse[i], i + test_start))
        # 显示统计信息
        st.subheader("异常检测统计信息结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("当前阈值", f"{valid_anomaly_max * alpha:.2f}")
        with col2:
            st.metric("检测到的异常数量", len(anomaly_collection))

        st.subheader("检测到的异常点")
        if anomaly_collection:
            anomaly_indices = [idx for _, idx in anomaly_collection]
            selected_idx = st.selectbox(
            "选择要查看热力图的异常点 (时间步)", anomaly_indices, format_func=lambda x: f"t={x}"
            )
            for error_matrix, idx in anomaly_collection:
                if idx == selected_idx:
                    fig_heatmap,root_cause_indices = plot_error_heatmap(error_matrix, idx)
                    st.pyplot(fig_heatmap)
                    st.write(f"异常点 t={idx} 的根因分析:")
                    st.write(f"Likely Root Cause Sensors: {root_cause_indices.tolist()}")
                    break
        else:
            st.info("未检测到异常点。")
            
    except Exception as e:
        st.error(f"加载数据时出错: {str(e)}")
        st.info("请确保数据文件路径正确，并且所有必要的文件都存在。")

if __name__ == "__main__":
    main() 