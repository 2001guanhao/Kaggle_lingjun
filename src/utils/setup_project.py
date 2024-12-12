"""项目初始化工具"""
import logging
from pathlib import Path
from .file_utils import ensure_dir

logger = logging.getLogger(__name__)

def setup_project():
    """初始化项目结构"""
    root = Path(__file__).parent.parent.parent
    
    # 创建目录结构
    dirs = [
        'data/raw',
        'data/processed',
        'data/output',
        'models',
        'logs',
        'predictions',
        'plots'  # 添加plots目录用于保存图表
    ]
    
    for d in dirs:
        dir_path = root / d
        ensure_dir(dir_path)
        logger.info(f"Created directory: {dir_path}")
    
    # 检查数据文件是否存在
    data_dir = root / 'data/raw'
    if not (data_dir / 'train.parquet').exists():
        logger.warning("训练数据文件不存在: data/raw/train.parquet")
    if not (data_dir / 'test.parquet').exists():
        logger.warning("测试数据文件不存在: data/raw/test.parquet")

if __name__ == "__main__":
    setup_project() 