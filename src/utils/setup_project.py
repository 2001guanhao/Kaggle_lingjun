"""项目初始化工具"""
import logging
from pathlib import Path
from .file_utils import ensure_dir
from .create_test_data import create_sample_data

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
        'predictions'
    ]
    
    for d in dirs:
        dir_path = root / d
        ensure_dir(dir_path)
        logger.info(f"Created directory: {dir_path}")
    
    # 生成示例数据
    data_dir = root / 'data/raw'
    if not (data_dir / 'train.csv').exists():
        logger.info("Generating sample data...")
        train_data, test_data = create_sample_data()
        train_data.to_csv(data_dir / 'train.csv', index=False)
        test_data.to_csv(data_dir / 'test.csv', index=False)
        logger.info("Sample data generated successfully")

if __name__ == "__main__":
    setup_project() 