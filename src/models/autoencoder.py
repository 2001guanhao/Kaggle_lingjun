"""自编码器模型"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
import gc

logger = logging.getLogger(__name__)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(train_data, test_data, device='mps'):
    """训练自编码器并返回降维后的数据"""
    logger.info("========== 自编码器训练开始 ==========")
    
    # 获取标签列和特征列
    label_cols = [f'Y{i}' for i in range(8)]
    feature_cols = [col for col in train_data.columns 
                   if col not in label_cols and not col.startswith('pred_')]
    
    # 准备数据
    train_features = train_data[feature_cols].values.astype(np.float32)
    test_features = test_data[feature_cols].values.astype(np.float32)
    train_labels = train_data[label_cols].values if label_cols[0] in train_data else None
    
    # 模型参数
    input_dim = train_features.shape[1]
    batch_size = 32
    model = AutoEncoder(input_dim=input_dim).to(device)
    
    # 训练设置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    train_dataset = TensorDataset(torch.FloatTensor(train_features))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 训练过程
    for epoch in range(50):
        total_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            encoded, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 清理内存
            del batch, encoded, decoded, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/50], Loss: {total_loss/len(train_loader):.4f}')
    
    # 生成编码结果
    model.eval()
    with torch.no_grad():
        train_encoded = []
        test_encoded = []
        
        # 训练集编码
        for i in range(0, len(train_features), batch_size):
            batch = torch.FloatTensor(train_features[i:i+batch_size]).to(device)
            encoded, _ = model(batch)
            train_encoded.append(encoded.cpu().numpy())
            
        # 测试集编码
        for i in range(0, len(test_features), batch_size):
            batch = torch.FloatTensor(test_features[i:i+batch_size]).to(device)
            encoded, _ = model(batch)
            test_encoded.append(encoded.cpu().numpy())
    
    # 合并结果
    train_encoded = np.concatenate(train_encoded)
    test_encoded = np.concatenate(test_encoded)
    
    # 转换为DataFrame
    encoded_cols = [f'encoded_{i}' for i in range(train_encoded.shape[1])]
    train_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=train_data.index)
    test_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=test_data.index)
    
    # 如果训练集有标签,添加到结果中
    if train_labels is not None:
        for i, col in enumerate(label_cols):
            train_df[col] = train_labels[:, i]
    
    logger.info("========== 自编码器训练完成 ==========")
    return train_df, test_df, model