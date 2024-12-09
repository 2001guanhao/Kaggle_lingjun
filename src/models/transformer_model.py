"""Transformer模型"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,  # 减小前馈网络大小
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

def select_features(train_data, n_features=100):
    """使用LightGBM选择重要特征"""
    feature_cols = [col for col in train_data.columns if not col.startswith('Y')]
    label_cols = [f'Y{i}' for i in range(8)]
    
    importance_dict = {}
    for col in feature_cols:
        importance_dict[col] = 0
    
    # 对每个目标变量训练LightGBM
    for target in label_cols:
        model = LGBMRegressor(n_estimators=100)
        model.fit(train_data[feature_cols], train_data[target])
        
        # 累加特征重要性
        for feat, imp in zip(feature_cols, model.feature_importances_):
            importance_dict[feat] += imp
    
    # 选择最重要的特征
    selected_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:n_features]
    return [feat[0] for feat in selected_features]

def train_and_predict(train_data, test_data, config, device='cpu'):
    """Transformer训练和预测"""
    logger.info("========== Transformer训练开始 ==========")
    
    # 选择重要特征
    selected_features = select_features(train_data)
    
    # 模型参数
    input_dim = len(selected_features)
    d_model = config.get('d_model', 128)  # 减小模型维度
    nhead = config.get('nhead', 4)  # 减少注意力头数
    num_layers = config.get('num_layers', 2)  # 减少层数
    output_dim = 8
    batch_size = config.get('batch_size', 32)  # 减小batch size
    n_epochs = config.get('epochs', 50)  # 减少训练轮数
    dropout = config.get('dropout', 0.1)
    
    # 准备数据
    X_train = torch.FloatTensor(train_data[selected_features].values).unsqueeze(1).to(device)
    y_train = torch.FloatTensor(train_data[[f'Y{i}' for i in range(8)]].values).to(device)
    X_test = torch.FloatTensor(test_data[selected_features].values).unsqueeze(1).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = TransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    best_loss = float('inf')
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
            
            # 清理内存
            del output, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}')
    
    # 分批生成预测以节省内存
    model.eval()
    train_preds = []
    test_preds = []
    
    with torch.no_grad():
        # 训练集预测
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size]
            pred = model(batch).cpu().numpy()
            train_preds.append(pred)
            del batch, pred
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # 测试集预测
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            pred = model(batch).cpu().numpy()
            test_preds.append(pred)
            del batch, pred
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    train_preds = np.concatenate(train_preds)
    test_preds = np.concatenate(test_preds)
    
    # 转换为DataFrame格式
    train_predictions = pd.DataFrame(
        train_preds,
        columns=[f'pred_transformer_Y{i}' for i in range(8)],
        index=train_data.index
    )
    
    test_predictions = pd.DataFrame(
        test_preds,
        columns=[f'pred_transformer_Y{i}' for i in range(8)],
        index=test_data.index
    )
    
    logger.info("========== Transformer训练完成 ==========")
    return train_predictions, test_predictions 