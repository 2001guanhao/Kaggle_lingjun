"""
机器学习模型集成主程序
"""
import os
import json
import yaml
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

from utils import setup_logging, get_device
from utils.data_processor import preprocess_data
from utils.visualization import (
    plot_training_history,
    plot_feature_importance,
    visualize_ensemble_weights
)
from models import train_ridge, train_lgbm, train_xgb, train_catboost
from models.ensemble import train_and_evaluate

logger = logging.getLogger(__name__)

def save_predictions(predictions, model_name):
    """保存模型预测结果
    
    Args:
        predictions: 预测结果DataFrame
        model_name: 模型名称
    """
    output_dir = Path(__file__).parent.parent / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保索引被正确保存
    predictions.index.name = 'ID'
    output_path = output_dir / f'{model_name}_predictions.parquet'
    predictions.to_parquet(output_path)
    logger.info(f"Saved {model_name} predictions to {output_path}")

def main(test_mode=False, n_samples=None):
    """主函数
    
    Args:
        test_mode: 是否为测试模式
        n_samples: 测试模式下使用的样本数
    """
    try:
        print("========== 1. 初始化 ==========")
        # 设置日志
        setup_logging()
        logger.info("Logging setup complete")
        
        # 加载配置
        config_path = Path(__file__).parent.parent / 'config' / 'model_config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded")
        
        print("\n========== 2. 加载数据 ==========")
        # 载数据
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        try:
            train_data = pd.read_parquet(data_dir / 'train.parquet')
            test_data = pd.read_parquet(data_dir / 'test.parquet')
        except FileNotFoundError:
            # 尝试上级目录
            data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
            train_data = pd.read_parquet(data_dir / 'train.parquet')
            test_data = pd.read_parquet(data_dir / 'test.parquet')
        
        print(f"数据目录: {data_dir}")
        print(f"训练集形状: {train_data.shape}")
        print(f"测试集形状: {test_data.shape}")
        
        # 检查索引是否已经设置
        if train_data.index.name != 'ID':
            train_data.set_index('ID', inplace=True)
        if test_data.index.name != 'ID':
            test_data.set_index('ID', inplace=True)
        
        # 打印设置索引后的信息
        print("\n数据集信息:")
        print(f"训练集索引: {train_data.index.name}")
        print(f"测试集索引: {test_data.index.name}")
        print("\n训练集列名:")
        print(train_data.columns.tolist())
        print("\n训练集前5行索引:")
        print(train_data.index[:5])
        print("\n测试集前5行索引:")
        print(test_data.index[:5])
        
        print(f"\n训练集形状: {train_data.shape}")
        print(f"测试集形状: {test_data.shape}")
        print("\n测试集示例:")
        print(test_data.head())
        
        if test_mode:
            print("\n========== 测试模式 ==========")
            logger.info(f"Running in test mode with {n_samples} samples")
            print(f"使用 {n_samples} 个样本进行测试")
            # 保存测试模式下使用的索引
            test_index = test_data.head(n_samples).index.copy()
            # 截取数据
            train_data = train_data.head(n_samples)
            test_data = test_data.head(n_samples)
        else:
            # 完整模式下使用全部索引
            test_index = test_data.index.copy()
        
        print("\n========== 3. 数据预处理 ==========")
        # 数据预处理和特征工程
        print("开始处理训练集...")
        train_processed, scaler = preprocess_data(train_data, is_train=True)
        print(f"训练集处理完成，形状: {train_processed.shape}")
        
        print("开始处理测试集...")
        test_processed, _ = preprocess_data(test_data, scaler=scaler, is_train=False)
        print(f"测试集处理完成，形状: {test_processed.shape}")
        
        # 存储所有模型的预测结果
        train_predictions = {}
        test_predictions = {}
        
        # 存储训练历史
        training_history = {}
        
        print("\n========== 4. 模型训练 ==========")
        # Ridge回归
        print("\n----- 4.1 Ridge回归 -----")
        train_ridge_preds, test_ridge_preds, ridge_history = train_ridge(
            train_data,
            test_data,
            target_cols=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'],
            config=config
        )
        training_history['ridge'] = ridge_history
        train_predictions['ridge'] = train_ridge_preds
        test_predictions['ridge'] = test_ridge_preds
        save_predictions(test_ridge_preds, 'ridge')
        plot_training_history(ridge_history, 'ridge')
        print("Ridge模型训练完成")
        
        # 将Ridge预测结果添加到特征中
        train_features = train_processed.copy()
        test_features = test_processed.copy()
        for col in train_ridge_preds.columns:
            train_features[col] = train_ridge_preds[col]
            test_features[col] = test_ridge_preds[col]
        
        print("\n----- 4.2 LightGBM -----")
        train_lgbm_preds, test_lgbm_preds, lgbm_history = train_lgbm(
            train_data,
            test_data,
            target_cols=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'],
            config=config['lgbm']
        )
        training_history['lgbm'] = lgbm_history
        train_predictions['lgbm'] = train_lgbm_preds
        test_predictions['lgbm'] = test_lgbm_preds
        save_predictions(test_lgbm_preds, 'lgbm')
        plot_training_history(lgbm_history, 'lgbm')
        if lgbm_history and 'feature_importance' in lgbm_history:
            feature_importance = lgbm_history['feature_importance']
            try:
                plot_feature_importance(feature_importance, 'lightgbm')
            except Exception as e:
                logger.error(f"处理LightGBM特征重要性时出错: {str(e)}")
        print("LightGBM模型训练完成")
        
        print("\n----- 4.3 XGBoost -----")
        train_xgb_preds, test_xgb_preds, xgb_history = train_xgb(
            train_data,
            test_data,
            target_cols=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'],
            config=config['xgb']
        )
        training_history['xgb'] = xgb_history
        train_predictions['xgb'] = train_xgb_preds
        test_predictions['xgb'] = test_xgb_preds
        save_predictions(test_xgb_preds, 'xgb')
        plot_training_history(xgb_history, 'xgb')
        if xgb_history and 'feature_importance' in xgb_history:
            feature_importance = xgb_history['feature_importance']
            try:
                plot_feature_importance(feature_importance, 'xgboost')
            except Exception as e:
                logger.error(f"处理XGBoost特征重要性时出错: {str(e)}")
        print("XGBoost模型训练完成")
        
        print("\n----- 4.4 CatBoost -----")
        train_cat_preds, test_cat_preds, cat_history = train_catboost(
            train_data,
            test_data,
            target_cols=['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'],
            config=config['catboost']
        )
        training_history['catboost'] = cat_history
        train_predictions['catboost'] = train_cat_preds
        test_predictions['catboost'] = test_cat_preds
        save_predictions(test_cat_preds, 'catboost')
        plot_training_history(cat_history, 'catboost')
        if cat_history and 'feature_importance' in cat_history:
            feature_importance = cat_history['feature_importance']
            try:
                plot_feature_importance(feature_importance, 'catboost')
            except Exception as e:
                logger.error(f"处理CatBoost特征重要性时出错: {str(e)}")
        print("CatBoost模型训练完成")
        
        # 准备包含所有模型预测的特征集
        all_features_train = train_features.copy()
        all_features_test = test_features.copy()
        
        # 添加所有模型的预测结果作为特征
        for model_name, preds in train_predictions.items():
            for col in preds.columns:
                all_features_train[col] = preds[col]
                all_features_test[col] = test_predictions[model_name][col]
        
        print("\n----- 4.5 特征选择 LightGBM -----")
        train_lgbm2_preds, test_lgbm2_preds, lgbm2_history = train_lgbm(
            train_data=all_features_train,
            test_data=all_features_test,
            config=config['lgbm']  # 使用相同的参数
        )
        training_history['lgbm2'] = lgbm2_history
        train_predictions['lgbm2'] = train_lgbm2_preds
        test_predictions['lgbm2'] = test_lgbm2_preds
        save_predictions(test_lgbm2_preds, 'lgbm2')
        plot_training_history(lgbm2_history, 'lgbm2')
        print("特征选择 LightGBM 训练完成")
        
        print("\n========== 5. 集成学习 ==========")
        # 将预测结果字典转换为DataFrame
        train_preds_df = pd.concat([df for df in train_predictions.values()], axis=1)
        test_preds_df = pd.concat([df for df in test_predictions.values()], axis=1)
        
        final_predictions, ensemble_weights = train_and_evaluate(
            train_preds=train_preds_df,
            test_preds=test_preds_df,
            y_true=train_data[[f'Y{i}' for i in range(8)]],
            config=config['ensemble']
        )
        visualize_ensemble_weights(ensemble_weights)
        print("集成学习完成")
        
        print("\n========== 6. 保存结果 ==========")
        # 保存最终的测试集预测结果
        output_path = Path(__file__).parent.parent / 'predictions' / 'submission_ml.csv'
        # 确���索引被正确保存
        final_predictions.index.name = 'ID'
        final_predictions.to_csv(output_path)
        print(f"预测结果已保存到: {output_path}")
        print(f"预测结果形状: {final_predictions.shape}")
        print(f"预测结果索引范围: {final_predictions.index.min()} - {final_predictions.index.max()}")
        print("\n预测结果示例:")
        print(final_predictions.head())
        
        # 将训练历史中的 NumPy 数组转换为列表
        processed_history = {}
        for model_name, history in training_history.items():
            processed_history[model_name] = {}
            for key, value in history.items():
                if isinstance(value, np.ndarray):
                    processed_history[model_name][key] = value.tolist()
                elif isinstance(value, dict):
                    # 处理特征重要性字典
                    processed_history[model_name][key] = {
                        k: float(v) if isinstance(v, np.number) else v
                        for k, v in value.items()
                    }
                else:
                    processed_history[model_name][key] = value
        
        # 保存训练历史
        history_path = Path(__file__).parent.parent / 'models' / 'training_history_ml.json'
        with open(history_path, 'w') as f:
            json.dump(processed_history, f, indent=4)
        print(f"训练历史已保存到: {history_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to use in test mode")
    args = parser.parse_args()
    
    main(test_mode=args.test, n_samples=args.samples) 