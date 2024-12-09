"""GRU模型"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        # 只使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

def train_and_predict(train_data, test_data, config, device='cpu'):
    """GRU训练和预测"""
    logger.info("========== GRU训练开始 ==========")
    
    # 模型参数
    input_dim = train_data.shape[1] - 8  # 减去8个目标变量
    hidden_dim = config.get('hidden_dim', 128)
    num_layers = config.get('num_layers', 2)
    output_dim = 8
    batch_size = config.get('batch_size', 64)
    n_epochs = config.get('epochs', 100)
    dropout = config.get('dropout', 0.2)
    
    # 准备数据
    feature_cols = [col for col in train_data.columns if not col.startswith('Y')]
    label_cols = [f'Y{i}' for i in range(8)]
    
    X_train = torch.FloatTensor(train_data[feature_cols].values).unsqueeze(1).to(device)
    y_train = torch.FloatTensor(train_data[label_cols].values).to(device)
    X_test = torch.FloatTensor(test_data[feature_cols].values).unsqueeze(1).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = GRUModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # 训练模型
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # 生成预测
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).cpu().numpy()
        test_preds = model(X_test).cpu().numpy()
    
    # 转换为DataFrame格式
    train_predictions = pd.DataFrame(
        train_preds,
        columns=[f'pred_gru_Y{i}' for i in range(8)],
        index=train_data.index
    )
    
    test_predictions = pd.DataFrame(
        test_preds,
        columns=[f'pred_gru_Y{i}' for i in range(8)],
        index=test_data.index
    )
    
    logger.info("========== GRU训练完成 ==========")
    return train_predictions, test_predictions 