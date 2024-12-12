"""Transformer模型"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMRegressor
from utils import get_train_val_split
from models.base_model import TimeSeriesDataset, train_epoch, evaluate_model, save_checkpoint
import copy
from pathlib import Path
from datetime import datetime

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
        # 添加序列维度 [batch_size, input_dim] -> [batch_size, 1, input_dim]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, d_model]
        x = self.embedding(x)
        
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        x = self.transformer(x)
        
        # 取序列的平均值 [batch_size, seq_len, d_model] -> [batch_size, d_model]
        x = x.mean(dim=1)
        
        # [batch_size, d_model] -> [batch_size, output_dim]
        x = self.fc(x)
        return x

def select_features(train_data, n_features=100):
    """特征选择
    
    Args:
        train_data: 训练数据
        n_features: 选择的特征数量
    
    Returns:
        selected_features: 选择的特征列表
    """
    # 基础特征
    feature_cols = [col for col in train_data.columns if not col.startswith('Y')]
    label_cols = [f'Y{i}' for i in range(8)]
    
    importance_dict = {}
    for col in feature_cols:
        importance_dict[col] = 0
    
    # 1. LightGBM特征重要性
    for target in label_cols:
        model = LGBMRegressor(n_estimators=100)
        model.fit(train_data[feature_cols], train_data[target])
        for feat, imp in zip(feature_cols, model.feature_importances_):
            importance_dict[feat] += imp
            
    # 2. 相关性分析
    for target in label_cols:
        correlations = train_data[feature_cols].corrwith(train_data[target]).abs()
        for feat, corr in correlations.items():
            importance_dict[feat] += corr
            
    # 3. 方差分析
    variances = train_data[feature_cols].var()
    for feat, var in variances.items():
        importance_dict[feat] += var / variances.max()  # 归一化方差
    
    # 选择最重要的特征
    selected_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:n_features]
    logger.info(f"Selected top {n_features} features")
    return [feat[0] for feat in selected_features]

def train_and_predict(train_data, test_data, config, device='mps'):
    """Transformer训练和预测"""
    logger.info("\n" + "="*50)
    logger.info("Transformer训练开始")
    logger.info("="*50)
    
    # 打印初始数据大小
    logger.info(f"\n初始数据大小:")
    logger.info(f"训练集: {train_data.shape}")
    logger.info(f"测试集: {test_data.shape}")
    
    # 使用 XGBoost 特征重要性
    if 'feature_importance' in config:
        feature_importance = config['feature_importance']
        n_features = config.get('n_features', 100)
        selected_features = [feat for feat, _ in sorted(feature_importance.items(), 
                                                      key=lambda x: x[1], 
                                                      reverse=True)[:n_features]]
        logger.info(f"\n使用 XGBoost 特征重要性选择了前 {n_features} 个特征")
        logger.info(f"前5个特征: {selected_features[:5]}")
    else:
        n_features = config.get('n_features', 100)
        selected_features = select_features(train_data, n_features)
    
    # 划分训练集验证集
    train_subset, val_subset = get_train_val_split(train_data)
    logger.info(f"\n数据集划分:")
    logger.info(f"训练子集: {train_subset.shape}")
    logger.info(f"验证子集: {val_subset.shape}")
    
    # 准备数据
    X_train = torch.FloatTensor(train_subset[selected_features].values).to(device)
    y_train = torch.FloatTensor(train_subset[[f'Y{i}' for i in range(8)]].values).to(device)
    X_val = torch.FloatTensor(val_subset[selected_features].values).to(device)
    y_val = torch.FloatTensor(val_subset[[f'Y{i}' for i in range(8)]].values).to(device)
    
    logger.info(f"\n数据张量大小:")
    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # 创建数据加载器
    batch_size = min(config.get('batch_size', 32), len(X_train))
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
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    
    # 初始化模型
    d_model = config.get('d_model', 256)
    nhead = config.get('nhead', 8)
    if d_model % nhead != 0:
        d_model = (d_model // nhead) * nhead
        logger.warning(f"调整 d_model 为 {d_model} 以被 nhead={nhead} 整除")
    
    model = TransformerModel(
        input_dim=len(selected_features),
        d_model=d_model,
        nhead=nhead,
        num_layers=config.get('num_layers', 3),
        output_dim=8,
        dropout=config.get('dropout', 0.1)
    ).to(device)
    
    logger.info(f"\n模型配置:")
    logger.info(f"输入维度: {len(selected_features)}")
    logger.info(f"d_model: {d_model}")
    logger.info(f"nhead: {nhead}")
    logger.info(f"num_layers: {config.get('num_layers', 3)}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练过程
    best_val_loss = float('inf')
    best_model_path = None
    train_losses = []
    val_losses = []
    
    logger.info("\n开始训练:")
    for epoch in range(config.get('epochs', 50)):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{config.get('epochs', 50)} "
                          f"[{batch_idx}/{len(train_loader)} "
                          f"({100. * batch_idx / len(train_loader):.0f}%)] "
                          f"Loss: {loss.item():.6f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        
        epoch_train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_model_path = Path(__file__).parent.parent.parent / 'models' / f'transformer_{timestamp}.pt'
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"\n保存最佳模型 - 验证损失: {val_loss:.6f}")
        
        logger.info(f"\nEpoch {epoch+1}/{config.get('epochs', 50)} - "
                  f"训练损失: {epoch_train_loss:.6f}, "
                  f"验证损失: {val_loss:.6f}")
    
    # 加载最佳模型进行预测
    logger.info(f"\n加载最佳模型: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # 预测
    with torch.no_grad():
        # 训练集预测
        train_preds = []
        X_full_train = torch.FloatTensor(train_data[selected_features].values).to(device)
        for i in range(0, len(train_data), batch_size):
            batch = X_full_train[i:i+batch_size]
            pred = model(batch).cpu().numpy()
            train_preds.append(pred)
            if i % (batch_size * 10) == 0:
                logger.info(f"训练集预测进度: {i}/{len(train_data)}")
        train_preds = np.concatenate(train_preds, axis=0)
        
        # 测试集预测
        test_preds = []
        X_test = torch.FloatTensor(test_data[selected_features].values).to(device)
        for i in range(0, len(test_data), batch_size):
            batch = X_test[i:i+batch_size]
            pred = model(batch).cpu().numpy()
            test_preds.append(pred)
            if i % (batch_size * 10) == 0:
                logger.info(f"测试集预测进度: {i}/{len(test_data)}")
        test_preds = np.concatenate(test_preds, axis=0)
    
    logger.info(f"\n预测结果形状:")
    logger.info(f"训练集预测: {train_preds.shape}")
    logger.info(f"测试集预测: {test_preds.shape}")
    
    # 创建预测结果DataFrame
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
    
    logger.info("\n预测完成")
    logger.info("="*50)
    
    return train_predictions, test_predictions, {
        'train_loss': train_losses,
        'val_loss': val_losses
    }