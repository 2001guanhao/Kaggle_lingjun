import logging
import yaml
import torch
import pandas as pd
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import traceback
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

from utils import setup_logging, get_device
from models import (
    train_ridge,
    train_lgbm,
    train_xgb,
    train_gru,
    train_transformer,
    train_and_evaluate,
    train_catboost
)
from utils.data_processor import load_data, preprocess_data, validate_data
from utils.setup_project import setup_project
from utils.save_utils import save_predictions
from utils.visualization import plot_training_history, plot_feature_importance
from utils.ensemble_utils import visualize_ensemble_weights

# 获取logger
logger = logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / 'config' / 'model_config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

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

def convert_numpy_types(obj):
    """递归转换所有NumPy类型"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main(test_mode=False, n_samples=None, ml_only=False):
    start_time = time.time()
    
    # 初始化字典用于存储结果
    training_history = {}
    train_predictions = {}
    test_predictions = {}
    
    print("\n========== 1. 初始化设置 ==========")
    # 确保项目结构和数据存在
    setup_project()
    
    # 加载配置
    config = load_config()
    print("配置加载完成")
    
    # 设置日志
    log_path = Path(__file__).parent.parent / 'logs' / 'train.log'
    logger = setup_logging(
        log_file=str(log_path),
        log_level='INFO'
    )

    # 确保日志正常工作
    logger.info("=" * 50)
    logger.info("开始新的训练运行")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    # 设置设备
    device = get_device(config['general']['device'])
    logger.info(f"使用设备: {device}")
    
    try:
        print("\n========== 2. 加载数据 ==========")
        # 加载数据
        train_data = load_data('data/raw/train.parquet')
        test_data = load_data('data/raw/test.parquet')
        
        # 验证数据
        validate_data(train_data, is_train=True)
        validate_data(test_data, is_train=False)
        
        assert train_data.index.name == 'ID', "训练集索引必须为'ID'"
        assert test_data.index.name == 'ID', "测试集索引必须为'ID'"
        assert all(f'Y{i}' in train_data.columns for i in range(8)), "训练集缺少目标变量"
        
        print(f"训练集形状: {train_data.shape}")
        print(f"测试集形状: {test_data.shape}")
        
        # 加载数据后立��打印
        print("\n原始数据范围:")
        print("\n训练集特征范围:")
        feature_cols = [col for col in train_data.columns if col.startswith('X')]
        target_cols = [f'Y{i}' for i in range(8)]
        for col in feature_cols[:5]:  # 只打印前5个特征
            print(f"{col} 范围: [{train_data[col].min():.4f}, {train_data[col].max():.4f}]")

        print("\n训练集目标变量范围:")
        for col in target_cols:
            print(f"{col} 范围: [{train_data[col].min():.4f}, {train_data[col].max():.4f}]")

        print("\n测试集特征范围:")
        for col in feature_cols[:5]:
            print(f"{col} 范围: [{test_data[col].min():.4f}, {test_data[col].max():.4f}]")
        
        # 打原始数据信息
        print("\n原始数据信息:")
        print(f"训练集索引: {train_data.index.name}")
        print(f"测试集索引: {test_data.index.name}")
        print("\n训练集前5行索引:")
        print(train_data.index[:5])
        print("\n测试集前5行索引:")
        print(test_data.index[:5])
        
        # 设置索引
        if 'ID' in train_data.columns:
            print("\n训练集包含ID列，设置为索引")
            train_data.set_index('ID', inplace=True)
        if 'ID' in test_data.columns:
            print("\n测试集包含ID列，设置为索引")
            test_data.set_index('ID', inplace=True)
        
        # 打印设置索引后的信息
        print("\n设置索引后的数据信息:")
        print(f"训练集索引: {train_data.index.name}")
        print(f"测试集索引: {test_data.index.name}")
        print("\n训练集前5行索引:")
        print(train_data.index[:5])
        print("\n测试集前5行索引:")
        print(test_data.index[:5])
        
        print(f"训练集形状: {train_data.shape}")
        print(f"测试集形状: {test_data.shape}")
        print(test_data.head())
        if test_mode:
            print("\n========== 测试模式 ==========")
            logger.info(f"使用前 {n_samples} 行数据进行测试")
            
            # 使用前n_samples行数据
            train_data = train_data.head(n_samples)
            test_data = test_data.head(n_samples)
            
            print(f"测试集训练数据形状: {train_data.shape}")
            print(f"测试集测试数据形状: {test_data.shape}")
            print("\n训练数据示例:")
            print(train_data.head())
            print("\n测试数据示例:")
            print(test_data.head())
            
            # 检查特征和标签
            feature_cols = [col for col in train_data.columns if col.startswith('X')]
            target_cols = [f'Y{i}' for i in range(8)]
            print(f"\n特征数量: {len(feature_cols)}")
            print(f"标签数量: {len(target_cols)}")
            
            # 检查数据类型
            print("\n数据类型:")
            print(train_data.dtypes.value_counts())
            
            # 检查缺失值
            print("\n缺失值统计:")
            print(train_data.isnull().sum().sum())
        else:
            # 完整模式下使用全部索引
            test_index = test_data.index.copy()
        
        print("\n========== 3. 特征工程 ==========")
        # 获取特征列
        feature_cols = [col for col in train_data.columns if col.startswith('X')]
        target_cols = [f'Y{i}' for i in range(8)]

        # 特征选择
        feature_selector = LGBMRegressor(
            objective='regression',
            metric='rmse',
            num_leaves=31,
            learning_rate=0.01,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbose=-1
        )

        # 计算每个特征的重要性
        feature_importance = {}
        for target in target_cols:
            feature_selector.fit(train_data[feature_cols], train_data[target])
            importance = dict(zip(feature_cols, feature_selector.feature_importances_))
            for feat, imp in importance.items():
                feature_importance[feat] = feature_importance.get(feat, 0) + imp

        # 选择前30个最重要的特征
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:30]
        selected_features = [feat[0] for feat in top_features]

        print("\n选择的前30个重要特征:")
        for i, (feat, imp) in enumerate(top_features, 1):
            print(f"{i:2d}. {feat}: {imp:.4f}")

        # 准备特征工程后的数据
        train_features = train_data.copy()
        test_features = test_data.copy()

        # 移除所有Y列
        for col in train_features.columns:
            if col.startswith('Y'):
                train_features.drop(col, axis=1, inplace=True)
                if col in test_features.columns:
                    test_features.drop(col, axis=1, inplace=True)

        # 在特征工程之前标准化原始特征
        print("\n对原始特征进行标准化")
        scaler = StandardScaler()
        for col in feature_cols:  # 只标准化X特征
            train_features[col] = scaler.fit_transform(train_features[[col]])
            test_features[col] = scaler.transform(test_features[[col]])

        print("原始特征标准化后的范围:")
        print(f"训练集: [{train_features[feature_cols].min().min():.4f}, {train_features[feature_cols].max().max():.4f}]")
        print(f"测试集: [{test_features[feature_cols].min().min():.4f}, {test_features[feature_cols].max().max():.4f}]")

        # 使用标准化后的数据进行特征交互
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                feat1, feat2 = selected_features[i], selected_features[j]
                # 乘积交互
                train_features[f'{feat1}_{feat2}_mul'] = train_features[feat1] * train_features[feat2]
                test_features[f'{feat1}_{feat2}_mul'] = test_features[feat1] * test_features[feat2]
                # 比值交互（注意除零）
                train_features[f'{feat1}_{feat2}_div'] = train_features[feat1] / (train_features[feat2] + 1e-8)
                test_features[f'{feat1}_{feat2}_div'] = test_features[feat1] / (test_features[feat2] + 1e-8)

        # 特征交互后再次标准化
        print("\n对交互特征进行标准化")
        for col in train_features.columns:
            if col not in feature_cols:  # 只标准化新生成的交互特征
                train_features[col] = scaler.fit_transform(train_features[[col]])
                test_features[col] = scaler.transform(test_features[[col]])

        print(f"\n特征工程后的特征数量: {train_features.shape[1]}")
        print("特征工程完成")
        
        # 检查目标变量是否存在
        print("\n检查目标变量:")
        for target in target_cols:
            if target in train_features.columns:
                print(f"{target} 存在")
            else:
                print(f"{target} 不存在!")
        
        print("\n========== 4. 模型训练 ==========")
        # Ridge回归
        print("\n----- 4.1 Ridge回归 -----")
        train_ridge_preds, test_ridge_preds, ridge_history = train_ridge(
            train_data=train_features,  # 使用最终标准化的特征
            test_data=test_features,
            target_cols=target_cols,
            config=config['ridge'],
            target_data=train_data[target_cols]
        )
        training_history['ridge'] = ridge_history
        train_predictions['ridge'] = train_ridge_preds
        test_predictions['ridge'] = test_ridge_preds
        print("Ridge模型训练完成")
        
        # LightGBM
        print("\n----- 4.2 LightGBM -----")
        train_lgbm_preds, test_lgbm_preds, lgbm_history = train_lgbm(
            train_data=train_features,  # 使用最终标准化的特征
            test_data=test_features,
            target_cols=target_cols,
            config=config['lgbm'],
            target_data=train_data[target_cols]
        )
        training_history['lgbm'] = lgbm_history
        train_predictions['lgbm'] = train_lgbm_preds
        test_predictions['lgbm'] = test_lgbm_preds
        print("LightGBM模型训练完成")
        
        # XGBoost
        print("\n----- 4.3 XGBoost -----")
        train_xgb_preds, test_xgb_preds, xgb_history = train_xgb(
            train_data=train_features,  # 使用最终标准化的特征
            test_data=test_features,
            target_cols=target_cols,
            config=config['xgb'],
            target_data=train_data[target_cols]
        )
        training_history['xgb'] = xgb_history
        train_predictions['xgb'] = train_xgb_preds
        test_predictions['xgb'] = test_xgb_preds
        print("XGBoost模型训练完成")
        
        # CatBoost
        print("\n----- 4.4 CatBoost -----")
        train_cat_preds, test_cat_preds, cat_history = train_catboost(
            train_data=train_features,  # 使用最终标准化的特征
            test_data=test_features,
            target_cols=target_cols,
            config=config['catboost'],
            target_data=train_data[target_cols]
        )
        training_history['catboost'] = cat_history
        train_predictions['catboost'] = train_cat_preds
        test_predictions['catboost'] = test_cat_preds
        print("CatBoost模型训练完成")
        
        # 特征选择和第二个LightGBM
        print("\n----- 4.5 特征选择 LightGBM -----")
        # 准备包含所有基础模型预测的特征
        all_features_train = train_data.copy()
        all_features_test = test_data.copy()

        # 添加所有基础模型的预测结果作为特征
        for model_name, preds in train_predictions.items():
            for col in preds.columns:
                all_features_train[col] = preds[col]
                all_features_test[col] = test_predictions[model_name][col]

        # 使用基础参数而不是优化
        base_lgbm_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'max_bin': 255,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'feature_pre_filter': False,
            'optimization': {
                'init_rounds': 1000,
                'early_stopping': 100
            }
        }

        # 训练第二个LGBM
        train_lgbm2_preds, test_lgbm2_preds, lgbm2_history = train_lgbm(
            train_data=all_features_train,
            test_data=all_features_test,
            target_cols=target_cols,
            config=base_lgbm_params,
            target_data=train_data[target_cols],
            do_optimize=False  # 不进行贝叶斯优化
        )

        # 保存第二个LGBM的结果
        training_history['lgbm2'] = lgbm2_history
        train_predictions['lgbm2'] = train_lgbm2_preds
        test_predictions['lgbm2'] = test_lgbm2_preds
        
        print("\n========== 5. 集成学习 ==========")
        # 将预测结果字典转换为DataFrame
        train_preds_df = pd.concat([df for df in train_predictions.values()], axis=1)
        test_preds_df = pd.concat([df for df in test_predictions.values()], axis=1)
        
        final_predictions, ensemble_weights = train_and_evaluate(
            train_preds=train_predictions,
            test_preds=test_predictions,
            target_cols=target_cols,
            train_data=train_data  # 使用原始训练数据，而不是特征数据
        )
        visualize_ensemble_weights(ensemble_weights)
        print("集成学习完成")
        
        print("\n========== 6. 保存结果 ==========")
        # 保存最终的测试集预测结果
        output_path = Path(__file__).parent.parent / 'predictions' / 'submission.csv'
        # 确保索引被正确保存
        final_predictions.index.name = 'ID'
        final_predictions.to_csv(output_path)
        print(f"预测结果已保存到: {output_path}")
        print(f"预测结果形状: {final_predictions.shape}")
        print(f"预测结果索引范围: {final_predictions.index.min()} - {final_predictions.index.max()}")
        print("\n预测结果示例:")
        print(final_predictions.head())
        
        # 保存每个模型的测试集预测结果
        test_predictions_path = Path(__file__).parent.parent / 'predictions' / 'test_predictions.csv'
        test_preds_df = pd.concat([df for df in test_predictions.values()], axis=1)
        test_preds_df.index.name = 'ID'
        test_preds_df.to_csv(test_predictions_path)
        print(f"\n各模型测试集预测结果已保存到: {test_predictions_path}")
        print(f"测试集预测结果形状: {test_preds_df.shape}")
        
        # 训���历史中的 NumPy 数组转换为列表
        processed_history = {}
        for model_name, history in training_history.items():
            processed_history[model_name] = convert_numpy_types(history)
        
        # 保存训练历史
        history_path = Path(__file__).parent.parent / 'models' / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(processed_history, f, indent=4)
        print(f"训练历史已保存到: {history_path}")
        
        # 计算总运行时间
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n总运行时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        
        # 在调用集成学习之前添加检查
        print("\n检查各模型预测结果:")
        for model_name, preds in train_predictions.items():
            print(f"\n{model_name}模型:")
            print(f"形状: {preds.shape}")
            print("列名:", preds.columns.tolist())
            if preds.isna().any().any():
                print("存在NaN值的列:")
                print(preds.isna().sum()[preds.isna().sum() > 0])
                print("NaN值所在行索引:")
                for col in preds.columns:
                    if preds[col].isna().any():
                        print(f"{col}: {preds[col][preds[col].isna()].index.tolist()}")

        # 检查lgbm和lgbm2的预测结果是否正确生成
        print("\n检查LGBM和LGBM2的预测过程:")
        print("LGBM预测形状:", train_lgbm_preds.shape)
        print("LGBM2预测形状:", train_lgbm2_preds.shape)
        
        # 在训练每个模型后，保存训练集的预测结果
        train_predictions['ridge'] = train_ridge_preds
        train_predictions['lgbm'] = train_lgbm_preds
        train_predictions['xgb'] = train_xgb_preds
        train_predictions['catboost'] = train_cat_preds
        train_predictions['lgbm2'] = train_lgbm2_preds

        # 打印预测结果的形状
        print("\n各模型训练集预测结果形状:")
        for model, preds in train_predictions.items():
            print(f"{model}: {preds.shape}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples for test mode')
    parser.add_argument('--ml-only', action='store_true', help='Run only machine learning models')
    args = parser.parse_args()
    
    main(test_mode=args.test, n_samples=args.samples, ml_only=args.ml_only) 