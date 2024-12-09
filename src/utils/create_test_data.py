"""示例数据生成工具"""
import numpy as np
import pandas as pd
from pathlib import Path

def create_sample_data(n_samples=1000, n_features=20):
    """创建示例数据集用于测试
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
    
    Returns:
        train_data: 训练集DataFrame
        test_data: 测试集DataFrame
    """
    # 生成特征
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    feature_cols = [f'X{i}' for i in range(n_features)]
    
    # 生成目标变量
    Y = np.zeros((n_samples, 8))
    for i in range(8):
        # 创建一些非线性关系
        Y[:, i] = (
            0.3 * X[:, i % n_features] + 
            0.4 * X[:, (i + 1) % n_features]**2 +
            0.3 * np.sin(X[:, (i + 2) % n_features]) +
            np.random.randn(n_samples) * 0.1
        )
    
    label_cols = [f'Y{i}' for i in range(8)]
    
    # 创建训练集
    train_data = pd.DataFrame(
        np.hstack([X, Y]),
        columns=feature_cols + label_cols
    )
    
    # 创建测试集（不包含目标变量）
    test_data = pd.DataFrame(
        np.random.randn(n_samples // 2, n_features),
        columns=feature_cols
    )
    
    return train_data, test_data

def save_sample_data():
    """保存示例数据到data/raw目录"""
    train_data, test_data = create_sample_data()
    
    # 确保目录存在
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据
    train_data.to_csv(data_dir / 'train.csv', index=False)
    test_data.to_csv(data_dir / 'test.csv', index=False)
    print(f"Sample data saved to {data_dir}")

if __name__ == "__main__":
    save_sample_data()
