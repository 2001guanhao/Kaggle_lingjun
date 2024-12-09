"""集成模型模块"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)

def train_and_evaluate(train_preds, test_preds, y_true, config):
    """使用线性回归训练集成模型并评估
    
    Args:
        train_preds: 训练集预测结果DataFrame
        test_preds: 测试集预测结果DataFrame
        y_true: 真实标签DataFrame
        config: 配置参数
    
    Returns:
        final_predictions: 最终预测结果DataFrame
    """
    logger.info("Training ensemble model...")
    
    # 获取每个模型的预测列
    model_types = ['catboost', 'gru', 'transformer', 'ridge', 'lgbm', 'xgb']
    pred_cols = {
        model: [col for col in train_preds.columns if col.startswith(f'pred_{model}_')]
        for model in model_types
    }
    
    # 优化每个目标的权重
    final_predictions = pd.DataFrame(index=test_preds.index)
    weights = {}
    
    for i in range(8):
        target = f'Y{i}'
        # 获取当前目标的所有模型预测
        X_train = pd.DataFrame({
            model: train_preds[pred_cols[model][i]].values
            for model in model_types if pred_cols[model]
        })
        
        # 训练线性回归模型
        lr_model = LinearRegression(positive=True)  # 使用正约束确保权重非负
        lr_model.fit(X_train, y_true[target])
        
        # 保存权重
        weights[target] = dict(zip(model_types, lr_model.coef_))
        
        # 生成测试集预测
        X_test = pd.DataFrame({
            model: test_preds[pred_cols[model][i]].values
            for model in model_types if pred_cols[model]
        })
        final_predictions[target] = lr_model.predict(X_test)
        
        # 记录权重
        weight_str = " ".join([f"{m}: {w:.3f}" for m, w in weights[target].items()])
        logger.info(f"Target {target} weights: {weight_str}")
    
    return final_predictions 