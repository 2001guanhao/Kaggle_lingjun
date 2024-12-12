import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def visualize_ensemble_weights(weights, save_dir='plots'):
    """可视化集成权重
    
    Args:
        weights: dict, 每个目标变量的模型权重
    """
    plt.figure(figsize=(12, 8))
    
    # 创建权重矩阵
    weight_df = pd.DataFrame(weights)
    
    # 热力图
    sns.heatmap(weight_df, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Model Ensemble Weights')
    plt.xlabel('Target Variable')
    plt.ylabel('Model')
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'ensemble_weights.png')
    plt.close() 