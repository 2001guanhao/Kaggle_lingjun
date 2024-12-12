"""集成模型模块"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

def train_and_evaluate(train_preds, test_preds, target_cols, train_data):
    """训练和评估集成模型"""
    print("\n========== 5. 集成学习 ==========")
    
    # 创建验证集 - 使用最后20%的数据
    val_size = int(len(train_data) * 0.2)
    train_subset = train_data.iloc[:-val_size]
    val_subset = train_data.iloc[-val_size:]
    
    print(f"\n训练集大小: {len(train_subset)}")
    print(f"验证集大小: {len(val_subset)}")
    
    # 打印可用的模型和它们的预测列
    print("\n可用的预测列:")
    for model, preds in train_preds.items():
        print(f"{model}的列: {preds.columns.tolist()}")
        # 检查预测值的范围
        for col in preds.columns:
            train_vals = preds[col].iloc[:-val_size]
            val_vals = preds[col].iloc[-val_size:]
            print(f"{col} 训练集范围: [{train_vals.min():.4f}, {train_vals.max():.4f}]")
            print(f"{col} 验证集范围: [{val_vals.min():.4f}, {val_vals.max():.4f}]")
    
    # 获取所有可用的模型
    available_models = list(train_preds.keys())
    print(f"\n可用的模型: {available_models}")
    
    final_predictions = pd.DataFrame(index=test_preds['ridge'].index)
    ensemble_weights = {}
    
    for target in target_cols:
        # 准备特征矩阵
        train_features = pd.DataFrame()
        val_features = pd.DataFrame()
        test_features = pd.DataFrame()
        
        # 收集各个模型的预测结果
        for model in available_models:
            pred_col = f'pred_{model}_{target}'
            if pred_col in train_preds[model].columns:
                train_features[model] = train_preds[model][pred_col].iloc[:-val_size]
                val_features[model] = train_preds[model][pred_col].iloc[-val_size:]
                test_features[model] = test_preds[model][pred_col]
                print(f"成功添加 {model} 的预测结果")
        
        # 使用Ridge回归在验证集上训练集成权重
        ensemble = Ridge(alpha=0.000001, positive=True)  # 添加正约束
        ensemble.fit(val_features, val_subset[target])  # 在验证集上训练
        
        # 归一化权重
        weights = ensemble.coef_
        weights = weights / np.sum(weights)  # 归一化使和为1
        weights_dict = dict(zip(val_features.columns, weights))
        ensemble_weights[target] = weights_dict
        
        # 使用归一化的权重进行预测
        final_pred = np.zeros(len(test_features))
        for model, weight in weights_dict.items():
            final_pred += weight * test_features[model]
        
        final_predictions[target] = final_pred
        
        # 计算训练集和验证集的RMSE
        train_pred = np.zeros(len(train_features))
        val_pred = np.zeros(len(val_features))
        for model, weight in weights_dict.items():
            train_pred += weight * train_features[model]
            val_pred += weight * val_features[model]
        
        train_rmse = np.sqrt(mean_squared_error(train_subset[target], train_pred))
        val_rmse = np.sqrt(mean_squared_error(val_subset[target], val_pred))
        
        # 打印权重和性能
        print(f"\n{target} 的模型权重 (和为1):")
        for model, weight in weights_dict.items():
            print(f"{model:>10}: {weight:>10.4f}")
        print(f"训练集RMSE: {train_rmse:.4f}")
        print(f"验证集RMSE: {val_rmse:.4f}")
        print("-" * 50)
    
    print("=" * 100)
    print("集成学习完成")
    
    return final_predictions, ensemble_weights 