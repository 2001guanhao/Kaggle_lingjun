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