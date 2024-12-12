import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict

def plot_training_history(history: Dict, model_name: str):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    
    if 'train_loss' in history and 'valid_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Valid Loss')
    
    plt.title(f'{model_name} Training History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    
    # 保存图片
    plt.savefig(f'plots/{model_name}_history.png')
    plt.close()

def plot_feature_importance(importance: Dict, model_name: str):
    """绘制特征重要性"""
    if not importance:
        return
        
    plt.figure(figsize=(12, 6))
    
    # 如果是列表（多个目标变量的特征重要性）
    if isinstance(importance, list):
        # 合并所有目标变量的特征重要性
        merged_importance = {}
        for imp_dict in importance:
            for feat, imp in imp_dict.items():
                merged_importance[feat] = merged_importance.get(feat, 0) + imp
        importance = merged_importance
    
    # 转换为Series并排序
    importance = pd.Series(importance)
    importance = importance.sort_values(ascending=True)[-20:]
    
    # 绘制条形图
    importance.plot(kind='barh')
    plt.title(f'{model_name} Feature Importance')
    plt.xlabel('Importance')
    
    # 保存图片
    plt.savefig(f'plots/{model_name}_importance.png')
    plt.close()

def visualize_ensemble_weights(weights, save_dir='plots'):
    """可视化集成权重
    
    Args:
        weights: dict, 每个目标变量的模型权重
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    weight_df = pd.DataFrame(weights)
    
    # 热力图
    sns.heatmap(weight_df, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Model Ensemble Weights')
    plt.xlabel('Target Variable')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(save_path / 'ensemble_weights.png')
    plt.close()

def plot_predictions(y_true, y_pred, model_name, target_col, save_dir='plots'):
    """绘制预测结果对比
    
    Args:
        y_true: Series, 真实值
        y_pred: Series, 预测值
        model_name: str, 模型名称
        target_col: str, 目标变量名
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f'{model_name} Predictions vs True Values ({target_col})')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.tight_layout()
    plt.savefig(save_path / f'{model_name}_{target_col}_predictions.png')
    plt.close() 