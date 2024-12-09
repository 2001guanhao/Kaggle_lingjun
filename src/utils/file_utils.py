"""文件和日志处理工具"""
import os
import logging
import sys
from pathlib import Path

def ensure_dir(dir_path):
    """确保目录存在"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent

def setup_logger(name='kaggle_competition', log_file=None):
    """配置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_dir = Path(log_file).parent
        ensure_dir(log_dir)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 