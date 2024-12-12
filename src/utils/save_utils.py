import pickle
import json
import pandas as pd
from pathlib import Path

def save_predictions(predictions, model_name, output_dir='predictions'):
    """保存模型预测结果
    
    Args:
        predictions: DataFrame, 预测结果
        model_name: str, 模型名称
        output_dir: str, 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 确保索引被正确保存
    predictions.index.name = 'ID'  # 设置索引名称
    
    # 保存为parquet格式，确保包含索引
    predictions.to_parquet(
        output_path / f'{model_name}_predictions.parquet',
        index=True  # 显式保存索引
    )

def load_predictions(model_name, output_dir='predictions'):
    """加载模型预测结果"""
    output_path = Path(output_dir)
    return pd.read_parquet(output_path / f'{model_name}_predictions.parquet') 