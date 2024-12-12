"""GRU模型"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import get_train_val_split
from .base_model import save_checkpoint
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # 添加序列维度如果需要
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, features]
        
        # GRU层
        out, _ = self.gru(x)  # out: [batch_size, seq_len, hidden_dim]
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])  # [batch_size, output_dim]
        return out

class TimeSeriesGRUDataset(Dataset):
    def __init__(self, data, feature_cols, label_cols=None, seq_length=10):
        """时序数据集
        
        Args:
            data: DataFrame, 原始数据
            feature_cols: list, 特征列名
            label_cols: list, 标签列名(Y0-Y7)，如果是测试集则为None
            seq_length: int, 序列长度
        """
        self.data = data
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.seq_length = seq_length
        
        # 检查数据和特征列是否有效
        if len(data) == 0:
            raise ValueError("数据集为空")
        if not all(col in data.columns for col in feature_cols):
            missing_cols = [col for col in feature_cols if col not in data.columns]
            raise ValueError(f"数据集中缺少以下特征列: {missing_cols}")
        
    def __len__(self):
        length = len(self.data) - self.seq_length + 1
        if length <= 0:
            raise ValueError(f"数据长度({len(self.data)})小于序列长度({self.seq_length})")
        return length
        
    def __getitem__(self, idx):
        # 获取序列数据，shape: (seq_length, n_features)
        sequence = self.data[self.feature_cols].iloc[idx:idx+self.seq_length].values
        sequence = torch.FloatTensor(sequence)  # shape: (seq_length, n_features)
        
        if self.label_cols:
            # 获取序列末尾对应的标签
            label = self.data[self.label_cols].iloc[idx+self.seq_length-1].values
            label = torch.FloatTensor(label)
            return sequence, label
        return sequence

def train_and_predict(train_data, test_data, config, device='mps'):
    """GRU训练和预测"""
    logger.info("\n" + "="*50)
    logger.info("GRU训练开始")
    logger.info("="*50)
    
    # 打印初始数据大小
    logger.info(f"\n初始数据大小:")
    logger.info(f"训练集: {train_data.shape}")
    logger.info(f"测试集: {test_data.shape}")
    
    # 获取特征列
    feature_cols = [col for col in train_data.columns 
                   if not col.startswith('Y') and not col.startswith('pred_')]
    label_cols = [f'Y{i}' for i in range(8)]
    
    # 使用已有的 XGBoost 特征重要性
    xgb_importance = config.get('feature_importance', {})
    if xgb_importance and isinstance(xgb_importance, dict) and len(xgb_importance) > 0:
        logger.info("\n使用 XGBoost 特征重要性")
        valid_features = [feat for feat in xgb_importance.keys() if feat in feature_cols]
        if not valid_features:
            logger.warning("未找到有效的 XGBoost 重要特征，使用所有特征")
            feature_cols = feature_cols
        else:
            selected_features = sorted(
                [(feat, xgb_importance[feat]) for feat in valid_features],
                key=lambda x: x[1],
                reverse=True
            )[:100]
            feature_cols = [feat[0] for feat in selected_features]
            logger.info(f"选择了前 {len(feature_cols)} 个特征")
            logger.info(f"前5个特征: {feature_cols[:5]}")
    
    # 模型参数
    input_dim = len(feature_cols)
    hidden_dim = config.get('hidden_dim', 128)
    num_layers = config.get('num_layers', 2)
    output_dim = 8
    dropout = config.get('dropout', 0.2)
    batch_size = min(16, len(train_data))  # 动态设置batch_size
    n_epochs = config.get('epochs', 100)
    
    logger.info(f"\n模型配置:")
    logger.info(f"输入维度: {input_dim}")
    logger.info(f"隐藏层维度: {hidden_dim}")
    logger.info(f"层数: {num_layers}")
    logger.info(f"Dropout: {dropout}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"训练轮数: {n_epochs}")
    
    # 划分训练集和验证集
    train_size = int(len(train_data) * 0.8)
    train_subset = train_data.iloc[:train_size]
    val_subset = train_data.iloc[train_size:]
    
    logger.info(f"\n数据集划分:")
    logger.info(f"训练子集: {train_subset.shape}")
    logger.info(f"验证子集: {val_subset.shape}")
    
    # 准备数据
    X_train = torch.FloatTensor(train_subset[feature_cols].values).to(device)
    y_train = torch.FloatTensor(train_subset[label_cols].values).to(device)
    X_val = torch.FloatTensor(val_subset[feature_cols].values).to(device)
    y_val = torch.FloatTensor(val_subset[label_cols].values).to(device)
    
    logger.info(f"\n数据张量大小:")
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"y_train: {y_train.shape}")
    logger.info(f"X_val: {X_val.shape}")
    logger.info(f"y_val: {y_val.shape}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"\n数据加载器配置:")
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    
    # 初始化模型
    model = GRUModel(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # 训练过程
    best_val_loss = float('inf')
    best_model_path = None
    train_losses = []
    val_losses = []
    
    logger.info("\n开始训练:")
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs} "
                          f"[{batch_idx}/{len(train_loader)} "
                          f"({100. * batch_idx / len(train_loader):.0f}%)] "
                          f"Loss: {loss.item():.6f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        epoch_train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_model_path = Path(__file__).parent.parent.parent / 'models' / f'gru_{timestamp}.pt'
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"\n保存最佳模型 - 验证损失: {val_loss:.6f}")
        
        logger.info(f"\nEpoch {epoch+1}/{n_epochs} - "
                  f"训练损失: {epoch_train_loss:.6f}, "
                  f"验证损失: {val_loss:.6f}")
    
    # 加载最佳模型进行预测
    logger.info(f"\n加载最佳模型: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # 创建预测结果DataFrame
    train_predictions = pd.DataFrame(
        index=train_data.index,
        columns=[f'pred_gru_Y{i}' for i in range(8)]
    )
    test_predictions = pd.DataFrame(
        index=test_data.index,
        columns=[f'pred_gru_Y{i}' for i in range(8)]
    )
    
    with torch.no_grad():
        # 训练集预测
        for i in range(0, len(train_data), batch_size):
            if i % (batch_size * 10) == 0:
                logger.info(f"训练集预测进度: {i}/{len(train_data)}")
            batch_data = train_data[feature_cols].iloc[i:i+batch_size]
            sequence = torch.FloatTensor(batch_data.values).to(device)
            pred = model(sequence).cpu().numpy()
            for j, idx in enumerate(batch_data.index):
                for k in range(8):
                    train_predictions.loc[idx, f'pred_gru_Y{k}'] = pred[j, k]
        
        # 测试集预测
        for i in range(0, len(test_data), batch_size):
            if i % (batch_size * 10) == 0:
                logger.info(f"测试集预测进度: {i}/{len(test_data)}")
            batch_data = test_data[feature_cols].iloc[i:i+batch_size]
            sequence = torch.FloatTensor(batch_data.values).to(device)
            pred = model(sequence).cpu().numpy()
            for j, idx in enumerate(batch_data.index):
                for k in range(8):
                    test_predictions.loc[idx, f'pred_gru_Y{k}'] = pred[j, k]
    
    logger.info(f"\n预测结果形状:")
    logger.info(f"训练集预测: {train_predictions.shape}")
    logger.info(f"测试集预测: {test_predictions.shape}")
    
    logger.info("\n预测完成")
    logger.info("="*50)
    
    return train_predictions, test_predictions, {
        'train_loss': train_losses,
        'val_loss': val_losses
    } 