"""传统机器学习模型"""
import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from bayes_opt import BayesianOptimization
import gc
from typing import Dict, Tuple, Any
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
import warnings

logger = logging.getLogger(__name__)

def bayesian_optimize(train_data: pd.DataFrame, val_data: pd.DataFrame, model_type: str, 
                     param_bounds: Dict[str, Tuple[float, float]], target_col: str, 
                     base_params: Dict[str, Any], opt_config: Dict) -> Tuple[Dict[str, Any], Any]:
    """贝叶斯优化超参数"""
    # 获取特征列
    feature_cols = [col for col in train_data.columns if col != target_col]
    
    # 创建数据集
    train_set = lgb.Dataset(train_data[feature_cols], train_data[target_col])
    val_set = lgb.Dataset(val_data[feature_cols], val_data[target_col], reference=train_set)
    
    def evaluate_params(**params):
        try:
            # 用基础参数
            current_params = base_params.copy()
            
            # 确保某些参数为整数类型
            int_params = ['num_leaves', 'max_bin', 'min_data_in_leaf']
            for param in int_params:
                if param in params:
                    params[param] = int(max(100, min(params[param], 1000)))
            
            current_params.update(params)
            
            # 训练模型
            model = lgb.train(
                params=current_params,
                train_set=train_set,
                num_boost_round=opt_config['init_rounds'],
                valid_sets=[val_set],
                valid_names=['valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # 保存当前模型
            evaluate_params.current_model = model
            
            # 获取最佳迭代的验证集分数
            best_score = model.best_score['valid']['rmse']
            
            # 处理无效值
            if np.isnan(best_score) or np.isinf(best_score):
                logger.warning(f"无效的评估分数: {best_score}")
                return float('-inf')
            
            return -best_score  # 返回负RMSE
            
        except Exception as e:
            logger.error(f"参数评估出错: {str(e)}")
            return float('-inf')
    
    try:
        # 使用初始参数评估
        initial_params = {k: base_params[k] for k in param_bounds.keys()}
        initial_score = evaluate_params(**initial_params)
        if initial_score == float('-inf'):
            logger.error("初始参数评估失败")
            raise ValueError("初始参数评估失败")
        
        optimizer = BayesianOptimization(
            f=evaluate_params,
            pbounds=param_bounds,
            random_state=42
        )
        optimizer.probe(params=initial_params)
        
        # 优化过程
        best_rmse = float('inf')
        no_improve_count = 0
        best_model = evaluate_params.current_model
        
        for i in range(opt_config['max_trials'] - 1):
            try:
                optimizer.maximize(init_points=0, n_iter=1)
                
                current_rmse = -optimizer.max['target']
                if current_rmse < best_rmse and not np.isinf(current_rmse):
                    best_rmse = current_rmse
                    best_model = evaluate_params.current_model
                    no_improve_count = 0
                    logger.info(f"找到更好的参数，RMSE: {current_rmse:.6f}")
                else:
                    no_improve_count += 1
                    if no_improve_count >= 2:
                        logger.info("连续2次无改善，提前停止")
                        break
                    
            except StopIteration:
                logger.warning("优化队列为空，提前停止")
                break
        
        if best_model is None:
            raise ValueError("优化失败，未找到有效模型")
        
        # 使用最优参数进行最终训练
        final_params = optimizer.max['params']
        
        # 确保整数参数
        for param in ['num_leaves', 'max_bin', 'min_data_in_leaf']:
            if param in final_params:
                final_params[param] = int(final_params[param])
        
        final_params.update(base_params)
        
        # 设置余弦退火学习率
        final_params['learning_rate'] = float(final_params['learning_rate'])
        final_model = lgb.train(
            final_params,
            train_set,
            num_boost_round=opt_config['final_rounds'],
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=opt_config['early_stopping']),
                lgb.reset_parameter(
                    learning_rate=lambda iter: final_params['learning_rate'] * 
                    (1 + np.cos(np.pi * iter / opt_config['final_rounds'])) / 2
                )
            ]
        )
        
        return final_params, final_model
        
    except Exception as e:
        logger.error(f"优化过程出错: {str(e)}")
        raise

def train_ridge(train_data, test_data, target_cols, config, target_data):
    """训练Ridge回归模型"""
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.linalg')
    
    print("\n----- 4.1 Ridge回归 -----")
    
    # 获取特征列（排除目标变量）
    feature_cols = [col for col in train_data.columns if not col.startswith('Y')]
    print(f"使用特征数量: {len(feature_cols)}")
    
    # 初始化预测结果DataFrame
    train_preds = pd.DataFrame(index=train_data.index)
    test_preds = pd.DataFrame(index=test_data.index)
    history = {'params': {}, 'train_loss': [], 'valid_loss': []}
    
    # 固定alpha值
    alpha = config.get('alpha', 0.1)  # 默认使用0.1
    print(f"使用正则化强度 alpha = {alpha}")
    
    # 添加调试信息
    print("\n特征数据范围:")
    print(f"训练集特征范围: [{train_data[feature_cols].min().min():.4f}, {train_data[feature_cols].max().max():.4f}]")
    print(f"测试集特征范围: [{test_data[feature_cols].min().min():.4f}, {test_data[feature_cols].max().max():.4f}]")
    
    # 创建验证集
    val_size = int(len(train_data) * 0.2)
    train_subset = train_data.iloc[:-val_size]
    val_subset = train_data.iloc[-val_size:]
    
    for i, target in enumerate(target_cols):
        print(f"\n训练目标: {target} ({i+1}/{len(target_cols)})")
        try:
            # 打印目标变量范围
            print(f"目标变量范围: [{target_data[target].min():.4f}, {target_data[target].max():.4f}]")
            
            # 训练模型
            model = Ridge(alpha=alpha)
            model.fit(train_subset[feature_cols], target_data[target].iloc[:-val_size])
            
            # 生成预测
            train_pred = model.predict(train_data[feature_cols])
            val_pred = model.predict(val_subset[feature_cols])
            test_pred = model.predict(test_data[feature_cols])
            
            # 打印预测范围
            print(f"训练集预测范围: [{train_pred.min():.4f}, {train_pred.max():.4f}]")
            print(f"验证集预测范围: [{val_pred.min():.4f}, {val_pred.max():.4f}]")
            print(f"测试集预测范围: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
            
            # 保存预测结果
            train_preds[f'pred_ridge_{target}'] = train_pred
            test_preds[f'pred_ridge_{target}'] = test_pred
            
            # 计算训练集和验证集的MSE
            train_mse = mean_squared_error(target_data[target].iloc[:-val_size], train_pred[:-val_size])
            val_mse = mean_squared_error(target_data[target].iloc[-val_size:], val_pred)
            
            print(f"{target} - Train MSE: {train_mse:.6f}")
            print(f"{target} - Valid MSE: {val_mse:.6f}")
            
            # 记录损失和参数
            history['train_loss'].append(np.sqrt(train_mse))
            history['valid_loss'].append(np.sqrt(val_mse))
            history['params'][target] = {'alpha': alpha}
            
            print(f"完成目标 {target} 的训练")
            
        except Exception as e:
            print(f"训练 {target} 时出错: {str(e)}")
            raise e
    
    print("Ridge模型训练完成")
    return train_preds, test_preds, history

def train_lgbm(train_data, test_data, target_cols, config, target_data, do_optimize=True):
    """训练LightGBM模型"""
    print("\n----- 4.2 LightGBM -----")
    
    # 获取特征列（排除目标变量）
    feature_cols = [col for col in train_data.columns if not col.startswith('Y')]
    print(f"特征数量: {len(feature_cols)}")
    
    # 确保训练集和测试集有相同的特征
    train_features = set(feature_cols)
    test_features = set(test_data.columns)
    if train_features != test_features:
        print("警告: 训练集和测试集特征不一致!")
        print(f"训练集特有特征: {train_features - test_features}")
        print(f"测试集特有特征: {test_features - train_features}")
        # 只使用共同的特征
        feature_cols = list(train_features.intersection(test_features))
        feature_cols.sort()  # 确保特征顺序一致
        print(f"使用共同特征: {len(feature_cols)}")
    
    # 保特征列表，确保后续使用相同的特征
    train_features = feature_cols.copy()
    
    # 打印前几个特征名称，用于调试
    print("\n前10个特征:")
    for i, feat in enumerate(feature_cols[:10]):
        print(f"{i+1}. {feat}")
    
    # 初始化预测结果DataFrame
    train_preds = pd.DataFrame(index=train_data.index)
    test_preds = pd.DataFrame(index=test_data.index)
    history = {
        'feature_importance': [],
        'best_params': {},
        'train_loss': [],
        'valid_loss': []
    }
    
    # 基础参数
    base_params = {
        'objective': config['objective'],
        'metric': config['metric'],
        'num_leaves': config['num_leaves'],
        'learning_rate': config['learning_rate'],
        'feature_fraction': config['feature_fraction'],
        'bagging_fraction': config['bagging_fraction'],
        'bagging_freq': config['bagging_freq'],
        'max_bin': config['max_bin'],
        'min_data_in_leaf': config['min_data_in_leaf'],
        'lambda_l1': config['lambda_l1'],
        'lambda_l2': config['lambda_l2'],
        'verbose': config['verbose'],
        'feature_pre_filter': config['feature_pre_filter'],
        'predict_disable_shape_check': True
    }
    
    # 优化参数
    opt_config = config.get('optimization', {})
    param_space = opt_config.get('param_space', {})  # 从配置中获取参数搜索空间
    init_rounds = opt_config.get('init_rounds', 300)
    final_rounds = opt_config.get('final_rounds', 3000)
    early_stopping = opt_config.get('early_stopping', 100)
    max_trials = opt_config.get('max_trials', 10)
    
    for target in target_cols:
        print(f"\n训练目标: {target}")
        
        # 创建验证集（包含所有列）
        val_size = int(len(train_data) * 0.2)
        train_subset = train_data.iloc[:-val_size].copy()
        val_subset = train_data.iloc[-val_size:].copy()
        
        if do_optimize:
            # 贝叶斯优化
            best_params, best_model = bayesian_optimize_lgbm(
                train_subset,
                val_subset,
                target,
                base_params,
                param_space,
                opt_config,
                target_data.iloc[:-val_size],  # 添加目标变量数据
                target_data.iloc[-val_size:]   # 添加目标变量数据
            )
        else:
            # 直接训练
            train_set = lgb.Dataset(train_subset[train_features], target_data[target].iloc[:-val_size])
            val_set = lgb.Dataset(val_subset[train_features], target_data[target].iloc[-val_size:], reference=train_set)
            
            best_model = lgb.train(
                params=base_params,
                train_set=train_set,
                num_boost_round=config['optimization']['init_rounds'],
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=config['optimization']['early_stopping']),
                    lgb.log_evaluation(period=100)
                ]
            )
            best_params = base_params
        
        # 使用最优模型预测，确保使用相同的特征
        train_pred = best_model.predict(train_data[train_features])
        test_pred = best_model.predict(test_data[train_features])
        
        # 保存预测结果
        model_name = 'lgbm2' if not do_optimize else 'lgbm'
        train_preds[f'pred_{model_name}_{target}'] = train_pred
        test_preds[f'pred_{model_name}_{target}'] = test_pred
        
        # 打印预测结果统计
        print(f"\n预测结果统计:")
        print(f"训练集预测范围: [{train_pred.min():.4f}, {train_pred.max():.4f}]")
        
        # 计算训练集和验证集的RMSE
        train_rmse = np.sqrt(mean_squared_error(target_data[target], train_pred))
        val_pred = best_model.predict(val_subset[train_features])
        val_rmse = np.sqrt(mean_squared_error(target_data[target].iloc[-val_size:], val_pred))
        
        print(f"训练集RMSE: {train_rmse:.4f}")
        print(f"验证集RMSE: {val_rmse:.4f}")
        
        # 记录损失
        history['train_loss'].append(train_rmse)
        history['valid_loss'].append(val_rmse)
        
        # 记录特征重要性和参数
        importance = dict(zip(train_features, best_model.feature_importance()))
        history['feature_importance'].append(importance)
        history['best_params'][target] = best_params
        
        print(f"完成目标 {target} 的训练")
    
    return train_preds, test_preds, history

def train_xgb(train_data, test_data, target_cols, config, target_data):
    """训练XGBoost模型"""
    logger.info("\n----- 4.3 XGBoost -----")
    
    # 获取特征列（排除目标变量）
    feature_cols = [col for col in train_data.columns if not col.startswith('Y')]
    logger.info(f"特征数量: {len(feature_cols)}")
    
    # 初始化预测结果DataFrame
    train_preds = pd.DataFrame(index=train_data.index)
    test_preds = pd.DataFrame(index=test_data.index)
    history = {
        'feature_importance': [],
        'best_params': {},
        'train_loss': [],
        'valid_loss': []
    }
    
    # 基础参数
    base_params = {
        'objective': config['objective'],
        'eval_metric': config['eval_metric'],
        'max_depth': config['max_depth'],
        'eta': config['eta'],
        'subsample': config['subsample'],
        'colsample_bytree': config['colsample_bytree'],
        'min_child_weight': config['min_child_weight'],
        'lambda': config['lambda'],
        'alpha': config['alpha'],
        'verbosity': 1
    }
    logger.info(f"基础参数: {base_params}")
    
    # 优化参数
    opt_config = config.get('optimization', {})
    param_space = opt_config.get('param_space', {})
    init_rounds = opt_config.get('init_rounds', 3000)
    final_rounds = opt_config.get('final_rounds', 3000)
    early_stopping = opt_config.get('early_stopping', 100)
    max_trials = opt_config.get('max_trials', 10)
    no_improve_trials = opt_config.get('no_improve_trials', 3)
    
    # 只在开始��打印一次表头
    logger.info("\n|   iter    |   rmse    |   alpha   | colsam... |    eta    |  lambda   | max_depth | min_ch... | subsample |")
    logger.info("-" * 110)
    
    # 对每个目标变量训练
    for i, target in enumerate(target_cols):
        logger.info(f"\n训练目标: {target} ({i+1}/{len(target_cols)})")
        
        # 创建验证集
        val_size = int(len(train_data) * 0.2)
        train_subset = train_data.iloc[:-val_size]
        val_subset = train_data.iloc[-val_size:]
        logger.info(f"训练集形状: {train_subset.shape}")
        logger.info(f"验证集形状: {val_subset.shape}")
        
        # 准备数据
        dtrain = xgb.DMatrix(train_subset[feature_cols], target_data[target].iloc[:-val_size])
        dval = xgb.DMatrix(val_subset[feature_cols], target_data[target].iloc[-val_size:])
        
        def xgb_evaluate(**params):
            try:
                current_params = base_params.copy()
                # 确保整数参数
                if 'max_depth' in params:
                    params['max_depth'] = int(params['max_depth'])
                if 'min_child_weight' in params:
                    params['min_child_weight'] = int(params['min_child_weight'])
                current_params.update(params)
                
                # 不再���这里打印表头
                if not hasattr(xgb_evaluate, 'iter'):
                    xgb_evaluate.iter = 1
                
                # 训练模型
                model = xgb.train(
                    params=current_params,
                    dtrain=dtrain,
                    num_boost_round=300,  # 优化期间使用300轮
                    early_stopping_rounds=50,
                    evals=[(dval, 'valid')],
                    verbose_eval=False
                )
                
                # 获取最佳分数
                best_score = model.best_score
                if np.isinf(best_score) or np.isnan(best_score):
                    best_score = float('inf')
                    
                # 打印参数和RMSE
                param_str = f"| {xgb_evaluate.iter:<8d} | {best_score:^8.4f} |"
                for param in ['alpha', 'colsample_bytree', 'eta', 'lambda', 'max_depth', 'min_child_weight', 'subsample']:
                    val = current_params.get(param, 0)
                    val = float(val) if isinstance(val, (int, float)) else 0.0
                    param_str += f" {val:^8.4f} |"
                logger.info(param_str)
                
                xgb_evaluate.iter += 1
                xgb_evaluate.current_model = model  # 保存当前模型
                xgb_evaluate.current_params = current_params  # 保存当前参数
                
                return -best_score
                
            except Exception as e:
                logger.error(f"参数评估出错: {str(e)}")
                return float('-inf')
        
        # 贝叶斯优化
        logger.info("开始贝叶斯优化...")
        optimizer = BayesianOptimization(
            f=xgb_evaluate,
            pbounds=param_space,
            random_state=42
        )
        
        best_rmse = float('inf')
        no_improve_count = 0
        best_model = None
        best_params = None
        
        # 初始点
        optimizer.maximize(init_points=2, n_iter=0)
        
        # 优化过程
        for i in range(max_trials - 2):
            try:
                optimizer.maximize(init_points=0, n_iter=1)
                current_rmse = -optimizer.max['target']
                
                if current_rmse < best_rmse:
                    best_rmse = current_rmse
                    best_model = xgb_evaluate.current_model
                    best_params = xgb_evaluate.current_params
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= 3:  # 连续3次无改善则停止
                        logger.info(f"连续3次无改善，提前停止优化")
                        break
                        
            except Exception as e:
                logger.error(f"优化过程出错: {str(e)}")
                break
        
        if best_model is None:
            raise ValueError("优化失败，未找到有效模型")
        
        logger.info(f"最优参数: {best_params}")
        logger.info(f"最优RMSE: {best_rmse:.6f}")
        
        # 使用最优参数进行最终训练
        final_model = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=final_rounds,
            early_stopping_rounds=early_stopping,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            verbose_eval=100
        )
        
        # 生成预测
        train_pred = final_model.predict(xgb.DMatrix(train_data[feature_cols]))
        test_pred = final_model.predict(xgb.DMatrix(test_data[feature_cols]))
        
        # 保存预测结果
        train_preds[f'pred_xgb_{target}'] = train_pred
        test_preds[f'pred_xgb_{target}'] = test_pred
        
        # 保存特征重要性和参数
        importance = final_model.get_score(importance_type='gain')
        history['feature_importance'].append(importance)
        history['best_params'][target] = best_params
        
        # 计算并记录训练集和验证集MSE
        train_mse = mean_squared_error(target_data[target], train_pred)
        val_pred = final_model.predict(xgb.DMatrix(val_subset[feature_cols]))
        val_mse = mean_squared_error(target_data[target].iloc[-val_size:], val_pred)
        
        logger.info(f"{target} - Train MSE: {train_mse:.6f}, Valid MSE: {val_mse:.6f}")
        history['train_loss'].append(np.sqrt(train_mse))
        history['valid_loss'].append(np.sqrt(val_mse))
        
        logger.info(f"完成目标 {target} 的训练\n")
    
    logger.info("XGBoost模型训练完成")
    return train_preds, test_preds, history

def train_catboost(train_data, test_data, target_cols, config, target_data):
    """训练CatBoost模型"""
    print("\n----- 4.4 CatBoost -----")
    
    # 获取特征列（排除目标变量）
    feature_cols = [col for col in train_data.columns if not col.startswith('Y')]
    print(f"特征数量: {len(feature_cols)}")
    
    # 初始化预测结果DataFrame
    train_preds = pd.DataFrame(index=train_data.index)
    test_preds = pd.DataFrame(index=test_data.index)
    history = {
        'feature_importance': [],
        'params': {},
        'train_loss': [],
        'valid_loss': []
    }
    
    # 使用与LightGBM类似的参数
    base_params = {
        'iterations': 3000,
        'learning_rate': 0.01,
        'depth': 6,  # 对应num_leaves
        'l2_leaf_reg': 3.0,  # 对应lambda_l2
        'bagging_temperature': 0.8,  # 对应bagging_fraction
        'random_strength': 1,
        'verbose': False,
        'early_stopping_rounds': 100
    }
    print(f"使用参数: {base_params}")
    
    for target in target_cols:
        print(f"\n训练目标: {target}")
        
        # 创建验证集
        val_size = int(len(train_data) * 0.2)
        train_subset = train_data.iloc[:-val_size].copy()
        val_subset = train_data.iloc[-val_size:].copy()
        
        print(f"训练集形状: {train_subset[feature_cols].shape}")
        print(f"验证集形状: {val_subset[feature_cols].shape}")
        
        # 创建模型
        model = CatBoostRegressor(**base_params)
        
        # 训练模型
        model.fit(
            train_subset[feature_cols],
            target_data[target].iloc[:-val_size],
            eval_set=(val_subset[feature_cols], target_data[target].iloc[-val_size:]),
            verbose=100
        )
        
        # 生成预测
        train_pred = model.predict(train_data[feature_cols])
        test_pred = model.predict(test_data[feature_cols])
        
        # 保存预测结果
        train_preds[f'pred_catboost_{target}'] = train_pred
        test_preds[f'pred_catboost_{target}'] = test_pred
        
        # 计算RMSE
        train_rmse = np.sqrt(mean_squared_error(target_data[target], train_pred))
        val_pred = model.predict(val_subset[feature_cols])
        val_rmse = np.sqrt(mean_squared_error(target_data[target].iloc[-val_size:], val_pred))
        
        print(f"训练集RMSE: {train_rmse:.4f}")
        print(f"验证集RMSE: {val_rmse:.4f}")
        
        # 记录结果
        history['train_loss'].append(train_rmse)
        history['valid_loss'].append(val_rmse)
        history['feature_importance'].append(
            dict(zip(feature_cols, model.feature_importances_))
        )
        history['params'][target] = base_params.copy()
        
        print(f"完成目标 {target} 训练")
    
    print("\nCatBoost模型训练完成")
    return train_preds, test_preds, history

def bayesian_optimize_lgbm(train_data, val_data, target, base_params, param_space, opt_config, train_target, val_target):
    """LightGBM的贝叶斯优化"""
    logger.info("开始LightGBM贝叶斯优化...")
    
    # 获取特征列（排除目标变量）
    feature_cols = [col for col in train_data.columns if not col.startswith('Y')]
    # 确保目标变量是一维的
    train_y = train_target[target].values
    val_y = val_target[target].values
    
    train_set = lgb.Dataset(train_data[feature_cols], train_y)
    val_set = lgb.Dataset(val_data[feature_cols], val_y, reference=train_set)

    def evaluate(**params):
        try:
            current_params = base_params.copy()
            # 确保整数参数
            for param in ['num_leaves', 'max_bin', 'min_data_in_leaf']:
                if param in params:
                    params[param] = int(params[param])
            current_params.update(params)
            
            # 打印参数和表头
            if not hasattr(evaluate, 'iter'):
                logger.info("\n|   iter    |  target   |   rmse    | baggin... | featur... | lambda_l1 | lambda_l2 | learni... | min_da... | num_le... |")
                logger.info("-" * 120)
                evaluate.iter = 1
            
            # 训练模型
            model = lgb.train(
                params=current_params,
                train_set=train_set,
                num_boost_round=300,  # 优化期间使用300轮
                valid_sets=[val_set],
                valid_names=['valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # 获取最佳分数
            val_pred = model.predict(val_data[feature_cols])
            val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))  # 使用一维的目标变量
            
            # 保存当前模型和参数
            evaluate.current_model = model
            evaluate.current_params = current_params
            
            # 打印参数和RMSE
            param_str = f"| {evaluate.iter:<8d} | {target:^8s} | {val_rmse:^8.4f} |"
            for param in ['bagging_fraction', 'feature_fraction', 'lambda_l1', 'lambda_l2', 
                         'learning_rate', 'min_data_in_leaf', 'num_leaves']:
                param_str += f" {current_params.get(param, 0):^8.4f} |"
            logger.info(param_str)
            
            evaluate.iter += 1
            return -val_rmse
            
        except Exception as e:
            logger.error(f"参数评估出错: {str(e)}")
            return float('-inf')
    
    try:
        # 贝叶斯优化
        optimizer = BayesianOptimization(
            f=evaluate,
            pbounds=param_space,
            random_state=42
        )
        
        # 使用初始参数评估
        initial_params = {k: base_params[k] for k in param_space.keys() if k in base_params}
        logger.info(f"初始参数: {initial_params}")
        initial_score = evaluate(**initial_params)
        if initial_score == float('-inf'):
            logger.error("初始参数评估失败")
            raise ValueError("初始参数评估失败")
        
        best_score = float('inf')
        best_model = None
        best_params = None
        no_improve_count = 0
        
        # 优化过程
        for i in range(opt_config['max_trials']):
            try:
                optimizer.maximize(init_points=0, n_iter=1)
                
                current_score = -optimizer.max['target']
                if current_score < best_score:
                    best_score = current_score
                    best_model = evaluate.current_model
                    best_params = evaluate.current_params
                    no_improve_count = 0
                    logger.info(f"找到更好的参数，RMSE: {current_score:.6f}")
                else:
                    no_improve_count += 1
                    if no_improve_count >= 3:  # 连续3次无改善则停止
                        logger.info(f"连续3次无改善，提前停止")
                        break
                    
            except Exception as e:
                logger.error(f"优化过程出错: {str(e)}")
                break
        
        if best_model is None:
            raise ValueError("优化失败，未找到有效模型")
        
        logger.info(f"最优参数: {best_params}")
        logger.info(f"最优RMSE: {best_score:.6f}")
        
        return best_params, best_model
        
    except Exception as e:
        logger.error(f"优化过程出错: {str(e)}")
        raise

def log_params_and_score(params: dict, score: float, prefix: str = ""):
    """统一的参数和数日志格式"""
    logger.info(f"\n{prefix}参数配置:")
    for k, v in sorted(params.items()):
        logger.info(f"{k}: {v}")
    logger.info(f"RMSE: {score:.6f}")