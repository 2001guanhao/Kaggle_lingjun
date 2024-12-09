import logging
import yaml
import torch
import pandas as pd
from pathlib import Path
import argparse

from src.utils import setup_logging, get_device
from src.models import (
    train_autoencoder,
    train_catboost,
    train_gru,
    train_transformer,
    train_ridge,
    train_lgbm,
    train_xgb,
    train_and_evaluate
)
from src.utils.data_processor import load_data, preprocess_data
from src.utils.setup_project import setup_project

def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / 'config' / 'model_config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

def main(test_mode=False, n_samples=100):
    # 确保项目结构和数据存在
    setup_project()
    
    # 加载配置
    config = load_config()
    
    # 设置日志
    log_path = Path(__file__).parent.parent / 'logs' / 'train.log'
    setup_logging(log_file=str(log_path))
    logger = logging.getLogger(__name__)
    
    # 设置设备
    device = get_device(config['general']['device'])
    logger.info(f"Using device: {device}")
    
    try:
        # 加载数据
        logger.info("Loading data...")
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        train_path = data_dir / 'train.parquet'
        test_path = data_dir / 'test.parquet'
        train_data = pd.read_parquet(train_path)
        test_data = pd.read_parquet(test_path)
        
        if test_mode:
            logger.info(f"Running in test mode with {n_samples} samples")
            # 保存测试模式下使用的索引
            test_index = test_data.head(n_samples).index.copy()
            # 截取数据
            train_data = train_data.head(n_samples)
            test_data = test_data.head(n_samples)
        else:
            # 完整模式下使用全部索引
            test_index = test_data.index.copy()
        
        # 数据预处理
        logger.info("Preprocessing data...")
        train_processed, test_processed = train_data.copy(), test_data.copy()
        
        # 训练各个模型并获取预测结果
        logger.info("Training models...")
        
        # Ridge回归
        train_ridge_preds, test_ridge_preds = train_ridge(
            train_data=train_processed, 
            test_data=test_processed,
            config=config['ridge']
        )
        
        # 将Ridge预测结果添加到特征中
        for col in train_ridge_preds.columns:
            train_processed[col] = train_ridge_preds[col]
            test_processed[col] = test_ridge_preds[col]
        
        # LightGBM
        train_lgbm_preds, test_lgbm_preds = train_lgbm(
            train_data=train_processed, 
            test_data=test_processed,
            config=config['lgbm']
        )
        
        # 将LightGBM预测结果添加到特征中
        for col in train_lgbm_preds.columns:
            train_processed[col] = train_lgbm_preds[col]
            test_processed[col] = test_lgbm_preds[col]
        
        # XGBoost
        train_xgb_preds, test_xgb_preds = train_xgb(
            train_data=train_processed, 
            test_data=test_processed,
            config=config['xgb']
        )
        
        # 将XGBoost预测结果添加到特征中
        for col in train_xgb_preds.columns:
            train_processed[col] = train_xgb_preds[col]
            test_processed[col] = test_xgb_preds[col]
        
        # CatBoost
        train_cat_preds, test_cat_preds = train_catboost(
            train_data=train_processed, 
            test_data=test_processed,
            config=config['catboost']
        )
        
        # GRU
        train_gru_preds, test_gru_preds = train_gru(
            train_data=train_processed, 
            test_data=test_processed,
            config=config['gru'],
            device=device
        )
        
        # Transformer
        train_trans_preds, test_trans_preds = train_transformer(
            train_data=train_processed, 
            test_data=test_processed,
            config=config['transformer'],
            device=device
        )
        
        # 获取验证集的索引(最后20%的数据)
        val_idx = int(len(train_processed) * 0.8)
        val_data = train_processed.iloc[val_idx:]
        
        # 为每个目标变量找到最优权重
        final_predictions = pd.DataFrame(index=test_index)
        
        for i in range(8):
            # 验证集上的各模型预测结果
            val_preds = pd.DataFrame({
                'ridge': train_ridge_preds.iloc[val_idx:][f'pred_ridge_Y{i}'],
                'lgbm': train_lgbm_preds.iloc[val_idx:][f'pred_lgbm_Y{i}'],
                'xgb': train_xgb_preds.iloc[val_idx:][f'pred_xgb_Y{i}'],
                'catboost': train_cat_preds.iloc[val_idx:][f'pred_catboost_Y{i}'],
                'gru': train_gru_preds.iloc[val_idx:][f'pred_gru_Y{i}'],
                'transformer': train_trans_preds.iloc[val_idx:][f'pred_transformer_Y{i}']
            })
            
            # 真实值
            val_true = val_data[f'Y{i}']
            
            # 使用线性回归找到最优权重
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(positive=True)  # 使用正约束
            lr.fit(val_preds, val_true)
            
            # 记录权重
            weights = dict(zip(val_preds.columns, lr.coef_))
            logger.info(f"Weights for Y{i}: {weights}")
            
            # 对测试集进行预测
            test_preds = pd.DataFrame({
                'ridge': test_ridge_preds[f'pred_ridge_Y{i}'],
                'lgbm': test_lgbm_preds[f'pred_lgbm_Y{i}'],
                'xgb': test_xgb_preds[f'pred_xgb_Y{i}'],
                'catboost': test_cat_preds[f'pred_catboost_Y{i}'],
                'gru': test_gru_preds[f'pred_gru_Y{i}'],
                'transformer': test_trans_preds[f'pred_transformer_Y{i}']
            })
            
            final_predictions[f'Y{i}'] = lr.predict(test_preds)
        
        # 保存预测结果
        output_path = Path(__file__).parent.parent / 'predictions' / 'submission.csv'
        final_predictions.to_csv(output_path, index=True)
        logger.info(f"Predictions saved to {output_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples for test mode')
    args = parser.parse_args()
    
    main(test_mode=args.test, n_samples=args.samples) 