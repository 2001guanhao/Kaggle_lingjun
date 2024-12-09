"""CatBoost模型"""
import numpy as np
import pandas as pd
import logging
from catboost import CatBoostRegressor
from src.utils.data_processor import get_train_val_split

logger = logging.getLogger(__name__)

def train_and_predict(train_data, test_data, config, device=None):
    """CatBoost训练和预测"""
    logger.info("========== CatBoost训练开始 ==========")
    
    # 获取标签列
    label_cols = [f'Y{i}' for i in range(8)]
    feature_cols = [col for col in train_data.columns 
                   if col not in label_cols and not col.startswith('pred_')]
    
    # 划分训练集和验证集
    train_subset, val_subset = get_train_val_split(train_data)
    
    # 存储预测结果
    train_predictions = pd.DataFrame(index=train_data.index)
    test_predictions = pd.DataFrame(index=test_data.index)
    
    # 训练8个目标的模型
    for i in range(8):
        logger.info(f"Training model for Y{i}")
        model = CatBoostRegressor(
            iterations=config.get('iterations', 2000),
            learning_rate=config.get('learning_rate', 0.01),
            depth=config.get('depth', 6),
            verbose=100,
            early_stopping_rounds=config.get('early_stopping_rounds', 200)
        )
        
        # 训练模型
        model.fit(
            train_subset[feature_cols],
            train_subset[f'Y{i}'],
            eval_set=(val_subset[feature_cols], val_subset[f'Y{i}']),
            early_stopping_rounds=200,
            verbose=100
        )
        
        # 生成预测
        train_predictions[f'pred_catboost_Y{i}'] = model.predict(train_data[feature_cols])
        test_predictions[f'pred_catboost_Y{i}'] = model.predict(test_data[feature_cols])
        
        logger.info(f"Completed training for Y{i}")
    
    logger.info("========== CatBoost训练完成 ==========")
    return train_predictions, test_predictions