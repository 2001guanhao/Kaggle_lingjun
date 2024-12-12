from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import logging
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

def get_train_val_split(X, y, val_size=0.2):
    """获取训练集和验证集
    对于时间序列数据，使用最后val_size比例的数据作为验证集
    """
    val_idx = int(len(X) * (1 - val_size))
    X_train = X.iloc[:val_idx]
    X_val = X.iloc[val_idx:]
    y_train = y.iloc[:val_idx]
    y_val = y.iloc[val_idx:]
    return X_train, X_val, y_train, y_val

def optimize_lgbm_params(train_data, target_col, init_points=5, n_iter=25):
    """使用贝叶斯优化调整LightGBM参数"""
    train_subset, val_subset = get_train_val_split(train_data)
    feature_cols = [col for col in train_data.columns if not col.startswith('Y')]
    
    def lgb_evaluate(**params):
        params = {
            'num_leaves': int(params['num_leaves']),
            'learning_rate': params['learning_rate'],
            'feature_fraction': params['feature_fraction'],
            'bagging_fraction': params['bagging_fraction'],
            'min_data_in_leaf': int(params['min_data_in_leaf']),
            'lambda_l1': params['lambda_l1'],
            'lambda_l2': params['lambda_l2']
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            train_subset[feature_cols], 
            train_subset[target_col],
            eval_set=[(val_subset[feature_cols], val_subset[target_col])],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        y_pred = model.predict(val_subset[feature_cols])
        mse = np.mean((val_subset[target_col] - y_pred) ** 2)
        return -mse
    
    pbounds = {
        'num_leaves': (150, 300),
        'learning_rate': (0.005, 0.05),
        'feature_fraction': (0.6, 0.8),
        'bagging_fraction': (0.7, 0.9),
        'min_data_in_leaf': (800, 1200),
        'lambda_l1': (0.05, 0.2),
        'lambda_l2': (0.05, 0.2)
    }
    
    optimizer = BayesianOptimization(
        f=lgb_evaluate,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )
    
    logger.info("Starting LightGBM parameter optimization...")
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )
    
    best_params = optimizer.max['params']
    logger.info(f"Best LightGBM parameters found: {best_params}")
    return best_params

def optimize_xgb_params(train_data, target_col, init_points=5, n_iter=25):
    """使用贝叶斯优化调整XGBoost参数"""
    train_subset, val_subset = get_train_val_split(train_data)
    feature_cols = [col for col in train_data.columns if not col.startswith('Y')]
    
    def xgb_evaluate(**params):
        params = {
            'max_depth': int(params['max_depth']),
            'eta': params['eta'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'min_child_weight': int(params['min_child_weight']),
            'lambda': params['lambda'],
            'alpha': params['alpha']
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            train_subset[feature_cols], 
            train_subset[target_col],
            eval_set=[(val_subset[feature_cols], val_subset[target_col])],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        y_pred = model.predict(val_subset[feature_cols])
        mse = np.mean((val_subset[target_col] - y_pred) ** 2)
        return -mse
    
    pbounds = {
        'max_depth': (6, 10),
        'eta': (0.005, 0.05),
        'subsample': (0.7, 0.9),
        'colsample_bytree': (0.6, 0.8),
        'min_child_weight': (800, 1200),
        'lambda': (0.05, 0.2),
        'alpha': (0.05, 0.2)
    }
    
    optimizer = BayesianOptimization(
        f=xgb_evaluate,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )
    
    logger.info("Starting XGBoost parameter optimization...")
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )
    
    best_params = optimizer.max['params']
    logger.info(f"Best XGBoost parameters found: {best_params}")
    return best_params

def optimize_ridge_params(X, y, init_points=5, n_iter=25):
    """使用贝叶斯优化调整Ridge参数"""
    def ridge_evaluate(**params):
        model = Ridge(alpha=params['alpha'])
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return np.mean(scores)
    
    # 定义参数范围
    pbounds = {
        'alpha': (0.001, 10.0)
    }
    
    # 初始化优化器
    optimizer = BayesianOptimization(
        f=ridge_evaluate,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True  # 允许重复点
    )
    
    # 执行优化
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )
    
    return optimizer.max['params']

def optimize_catboost_params(train_data, target_col, init_points=5, n_iter=25):
    """使用贝叶斯优化调整CatBoost参数"""
    train_subset, val_subset = get_train_val_split(train_data)
    
    def catboost_evaluate(**params):
        params = {
            'depth': int(params['depth']),
            'learning_rate': params['learning_rate'],
            'l2_leaf_reg': params['l2_leaf_reg'],
            'random_strength': params['random_strength'],
            'bagging_temperature': params['bagging_temperature']
        }
        
        model = CatBoostRegressor(**params, verbose=False)
        model.fit(
            train_subset[feature_cols], 
            train_subset[target_col],
            eval_set=[(val_subset[feature_cols], val_subset[target_col])],
            early_stopping_rounds=50,
            verbose=False
        )
        y_pred = model.predict(val_subset[feature_cols])
        mse = np.mean((val_subset[target_col] - y_pred) ** 2)
        return -mse
    
    pbounds = {
        'depth': (6, 10),
        'learning_rate': (0.005, 0.05),
        'l2_leaf_reg': (0.05, 0.2),
        'random_strength': (0.05, 0.2),
        'bagging_temperature': (0.7, 0.9)
    }
    
    optimizer = BayesianOptimization(
        f=catboost_evaluate,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )
    
    logger.info("Starting CatBoost parameter optimization...")
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )
    
    best_params = optimizer.max['params']
    logger.info(f"Best CatBoost parameters found: {best_params}")
    return best_params