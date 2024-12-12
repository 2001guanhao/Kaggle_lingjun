from .file_utils import ensure_dir, get_project_root, setup_logger
from .data_processor import (
    load_data, 
    preprocess_data, 
    get_train_val_split
)
from .setup_project import setup_project

def get_device(device_name='auto'):
    """获取计算设备"""
    import torch
    if device_name == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_name

def setup_logging(log_file=None, log_level='INFO'):
    """设置日志配置"""
    from .file_utils import setup_logger
    return setup_logger(log_file=log_file) 

def get_train_val_split(df, val_size=0.2):
    """获取训练集和验证集
    对于时间序列数据，使用最后val_size比例的数据作为验证集
    
    Args:
        df: DataFrame, 包含特征和目标变量的数据集
        val_size: float, 验证集比例
        
    Returns:
        train_data: 训练集
        val_data: 验证集
    """
    val_idx = int(len(df) * (1 - val_size))
    train_data = df.iloc[:val_idx]
    val_data = df.iloc[val_idx:]
    return train_data, val_data 