"""传统机器学习模型"""
import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from ..utils.data_processor import get_train_val_split
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)

def train_ridge(train_data, test_data, config):
    """训练Ridge模型"""
    logger.info("========== Ridge训练开始 ==========")
    
    # 获取特征和标签列
    label_cols = [f'Y{i}' for i in range(8)]
    feature_cols = [col for col in train_data.columns 
                   if col not in label_cols and not col.startswith('pred_')]
    
    # 存储预测结果
    train_predictions = pd.DataFrame(index=train_data.index)
    test_predictions = pd.DataFrame(index=test_data.index)
    
    # 训练每个目标的模型
    for i in range(8):
        logger.info(f"Training Ridge model for Y{i}")
        model = Ridge(alpha=1.0)
        
        # 训练模型
        model.fit(train_data[feature_cols], train_data[f'Y{i}'])
        
        # 生成预测
        train_predictions[f'pred_ridge_Y{i}'] = model.predict(train_data[feature_cols])
        test_predictions[f'pred_ridge_Y{i}'] = model.predict(test_data[feature_cols])
    
    logger.info("========== Ridge训练完成 ==========")
    return train_predictions, test_predictions

def train_lgbm(train_data, test_data, config):
    """训练LightGBM模型"""
    logger.info("========== LightGBM训练开始 ==========")
    
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
        logger.info(f"Training LightGBM model for Y{i}")
        model = LGBMRegressor(**{
            'n_estimators': config.get('n_estimators', 1000),
            'learning_rate': config.get('learning_rate', 0.01),
            'num_leaves': config.get('num_leaves', 31),
            'max_depth': config.get('max_depth', -1),
            'min_child_samples': config.get('min_child_samples', 20),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8)
        })
        
        # 训练模型
        model.fit(
            train_subset[feature_cols],
            train_subset[f'Y{i}'],
            eval_set=[(val_subset[feature_cols], val_subset[f'Y{i}'])],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(100)]  # 使用callbacks替代early_stopping_rounds
        )
        
        # 生成预测
        train_predictions[f'pred_lgbm_Y{i}'] = model.predict(train_data[feature_cols])
        test_predictions[f'pred_lgbm_Y{i}'] = model.predict(test_data[feature_cols])
        
        logger.info(f"Completed training for Y{i}")
    
    logger.info("========== LightGBM训练完成 ==========")
    return train_predictions, test_predictions

def train_xgb(train_data, test_data, config):
    """训练XGBoost模型"""
    logger.info("========== XGBoost训练开始 ==========")
    
    # 获取标签列
    label_cols = [f'Y{i}' for i in range(8)]
    feature_cols = [col for col in train_data.columns 
                   if col not in label_cols and not col.startswith('pred_')]
    
    # 划分训练集和验证集
    train_subset, val_subset = get_train_val_split(train_data)
    
    # 存储预测结果
    train_predictions = pd.DataFrame(index=train_data.index)
    test_predictions = pd.DataFrame(index=test_data.index)
    
    # 训练每个目标的模型
    for i in range(8):
        logger.info(f"Training XGBoost model for Y{i}")
        model = XGBRegressor(**{
            'n_estimators': config.get('n_estimators', 1000),
            'learning_rate': config.get('learning_rate', 0.01),
            'max_depth': config.get('max_depth', 6),
            'min_child_weight': config.get('min_child_weight', 1),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
            'tree_method': config.get('tree_method', 'hist'),
            'early_stopping_rounds': 100  # 直接在参数中设置early_stopping
        })
        
        # 训练模型
        eval_set = [(val_subset[feature_cols], val_subset[f'Y{i}'])]
        model.fit(
            train_subset[feature_cols],
            train_subset[f'Y{i}'],
            eval_set=eval_set,
            verbose=True
        )
        
        # 生成预测
        train_predictions[f'pred_xgb_Y{i}'] = model.predict(train_data[feature_cols])
        test_predictions[f'pred_xgb_Y{i}'] = model.predict(test_data[feature_cols])
        
        logger.info(f"Completed training for Y{i}")
    
    logger.info("========== XGBoost训练完成 ==========")
    return train_predictions, test_predictions 