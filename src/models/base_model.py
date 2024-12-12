"""基础模型类"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, feature_cols, label_cols=None):
        """
        Args:
            data: DataFrame, 原始数据
            feature_cols: list, 特征列名
            label_cols: list, 标签列名(Y0-Y7)，如果是测试集则为None
        """
        self.data = data
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.data[self.feature_cols].iloc[idx].values)
        if self.label_cols:
            labels = torch.FloatTensor(self.data[self.label_cols].iloc[idx].values)
            return features, labels
        return features

def train_epoch(model, train_loader, optimizer, criterion, device, accumulation_steps=1):
    """优化的训练函数"""
    model.train()
    total_loss = 0
    
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        # 根据设备类型使用适当的 autocast
        if device == 'mps':
            output = model(data)
            loss = criterion(output, target)
        else:
            with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                output = model(data)
                loss = criterion(output, target)
        
        loss = loss / accumulation_steps  # 梯度累积
        
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # 清理内存
        del data, target, output, loss
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, criterion, device):
    """优化的评估函数"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            
            # 根据设备类型使用适当的 autocast
            if device == 'mps':
                output = model(data)
                loss = criterion(output, target)
            else:
                with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                    output = model(data)
                    loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # 清理内存
            del data, target, output, loss
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    return total_loss / len(val_loader)

def save_checkpoint(model, optimizer, epoch, loss, model_name, target_col=None, path=None):
    """保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        loss: 当前损失
        model_name: 模型名称
        target_col: 目标列名（对于单独训练的目标）
        path: 保存路径，如果为None则使用默认路径
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / 'models'
        path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名：model_name_target_timestamp.pt
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}"
        if target_col:
            filename += f"_{target_col}"
        filename += f"_{timestamp}.pt"
        path = path / filename
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Model saved to {path}")
    return path  # 返回保存路径，方便后续加载