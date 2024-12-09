"""数据处理工具"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def ensure_dir(dir_path):
    """确保目录存在,如果不存在则创建"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent

def setup_project_structure():
    """创建项目基本目录结构"""
    root = get_project_root()
    
    # 定义需要创建的目录
    dirs = [
        'data/raw',
        'data/processed', 
        'data/output',
        'logs',
        'config',
        'notebooks'
    ]
    
    # 创建目录
    for d in dirs:
        dir_path = root / d
        ensure_dir(dir_path)
        logger.info(f"Created directory: {dir_path}")

def load_data(data_path):
    """加载数据"""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def preprocess_data(df, scaler=None):
    """数据预处理"""
    # 分离特征和标签
    label_cols = [f'Y{i}' for i in range(8)]
    feature_cols = df.columns.difference(label_cols)
    
    X = df[feature_cols]
    y = df[label_cols] if all(col in df.columns for col in label_cols) else None
    
    # 标准化
    if scaler is None:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    else:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    
    return X, y, scaler

def get_train_val_split(df, val_size=0.2):
    """划分训练集和验证集"""
    train_size = int(len(df) * (1 - val_size))
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:]
    return train_data, val_data 