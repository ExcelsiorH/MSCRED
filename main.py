import torch
import torch.nn as nn
from tqdm import tqdm
from model.mscred import MSCRED
from utils.data import load_data
import numpy as np
import os

# 确保在 GPU 环境下
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train(dataLoader, model, optimizer, epochs, device):
    model = model.to(device)
    model.train()
    print("------training on {}-------".format(device))
    
    for epoch in range(epochs):
        train_l_sum, n = 0.0, 0
        for x in tqdm(dataLoader):
            # 假设 dataLoader 加载出的 batch_size 为 1, shape: (1, 5, 3, 15, 15)
            # 如果 batch_size > 1，需要 dataLoader 配合
            x = x.to(device)
            # x 不需要 squeeze 掉 batch 维，MSCRED 现在支持 (Batch, Step, C, H, W)
            # 目标是重建最后一个时间步的图
            target = x[:, -1, :, :, :] # Shape: (Batch, 3, 15, 15)
            
            output = model(x) # Output: (Batch, 3, 15, 15)
            
            l = torch.mean((output - target)**2)
            
            train_l_sum += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += 1
            
        print("[Epoch %d/%d] [loss: %f]" % (epoch+1, epochs, train_l_sum/n))

def test(dataLoader, model):
    print("------Testing-------")
    model.eval()
    index = 65 # 自定义起始索引
    reconstructed_data_path = "./data/matrix_data/reconstructed_data/"
    if not os.path.exists(reconstructed_data_path):
        os.makedirs(reconstructed_data_path)
        
    with torch.no_grad():
        for x in dataLoader:
            x = x.to(device)
            reconstructed_matrix = model(x) 
            # 保存每一个样本的重建结果
            # 如果 batch_size > 1，这里需要循环保存
            for i in range(x.shape[0]):
                save_path = os.path.join(reconstructed_data_path, 'reconstructed_data_' + str(index) + ".npy")
                np.save(save_path, reconstructed_matrix[i].cpu().detach().numpy())
                index += 1

if __name__ == '__main__':
    dataLoader = load_data() # 请确保 load_data 返回包含 'train' 和 'test' DataLoader 的字典
    
    # MSCRED(3, 256): 3个输入通道(窗口数)，256个最深层特征通道
    mscred = MSCRED(3, 256)

    # 训练阶段
    optimizer = torch.optim.Adam(mscred.parameters(), lr = 0.0002)
    train(dataLoader["train"], mscred, optimizer, 10, device)
    
    print("保存模型中....")
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    torch.save(mscred.state_dict(), "./checkpoints/model_1121.pth")

    # 测试阶段
    mscred.load_state_dict(torch.load("./checkpoints/model_1121.pth", weights_only=True))
    mscred.to(device)
    test(dataLoader["test"], mscred)