import logging
import sys
from pathlib import Path

def setup_logging(log_file=None, log_level='INFO'):
    """设置日志"""
    # 创建logger
    logger = logging.getLogger('kaggle_competition')  # 使用固定的logger名称
    logger.handlers.clear()
    
    # 设置日志级别
    level = getattr(logging, log_level.upper())
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保文件存在
        Path(log_file).touch(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')  # 使用追加模式
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    # 禁用传播到根logger
    logger.propagate = False
    
    return logger