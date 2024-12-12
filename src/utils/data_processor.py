"""数据预处理模块"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import logging
from pathlib import Path
import os
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)

# 保存全局的重要特征列表
IMPORTANT_FEATURES = None

def ensure_dir(dir_path):
    """确保目录存在,如果不存在则创建"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent

def setup_project_structure():
    """创建项目基本目录结构"""
    root = get_project_root()
    
    # 定义需要创建的目录
    dirs = [
        'data/raw',
        'data/processed', 
        'data/output',
        'logs',
        'config',
        'notebooks'
    ]
    
    # 创建目录
    for d in dirs:
        dir_path = root / d
        ensure_dir(dir_path)
        logger.info(f"Created directory: {dir_path}")

def load_data(data_path: str) -> pd.DataFrame:
    """加载数据并设置索引
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        df: 处理后的数据框
    """
    try:
        # 读取数据
        df = pd.read_parquet(data_path)
        
        # 检查并设置ID为索引
        if 'ID' in df.columns:
            df.set_index('ID', inplace=True)
        elif df.index.name != 'ID':
            raise ValueError("数据缺少ID列或索引")
            
        logger.info(f"数据加载完成，形状: {df.shape}")
        logger.info(f"索引范围: {df.index.min()} - {df.index.max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise

def get_train_val_split(df, val_size=0.2):
    """取训练集和验证集
    对于时间序列数据，使用最后val_size比例的数据作为验证集
    
    Args:
        df: DataFrame, 包含特征和目标变量的数据集
        val_size: float, 验证集比例
    
    Returns:
        train_data: 训练集
        val_data: 验证集
    """
    # 打印原始数据信息
    logger.info(f"划分前数据大小: {df.shape}")
    logger.info(f"划分前索引范围: {df.index.min()} - {df.index.max()}")
    
    # 如果数据集太小，返回完整数据集作为训练集和验证集
    if len(df) < 10:  # 设置一个最小阈值
        logger.warning("数据集太小，使用完整数据集作为训练集和验证集")
        return df, df
    
    val_idx = int(len(df) * (1 - val_size))
    train_data = df  # 返回完整数据集作为训练数据
    val_data = df.iloc[val_idx:]  # 验证集仍然使用后面的部分
    
    # 打印划分后的信息
    logger.info(f"划分后 - 训练集大小: {train_data.shape}, 验证集大小: {val_data.shape}")
    logger.info(f"训练集索引范围: {train_data.index.min()} - {train_data.index.max()}")
    logger.info(f"验证集索引范围: {val_data.index.min()} - {val_data.index.max()}")
    
    return train_data, val_data

def preprocess_data(data: pd.DataFrame, is_train: bool = True, scaler=None):
    """数据预处理"""
    # 1. 获取特征列
    feature_cols = [col for col in data.columns if col.startswith('X')]
    target_cols = [f'Y{i}' for i in range(8)] if is_train else []
    
    # 2. 处理原始特征的无穷值和极值
    processed_data = data.copy()
    numeric_cols = processed_data[feature_cols].select_dtypes(include=[np.number]).columns
    
    # 2.1 处理无穷值
    inf_mask = np.isinf(processed_data[numeric_cols])
    if inf_mask.any().any():
        logger.warning("发现无穷值，将替换为0")
        processed_data[numeric_cols] = processed_data[numeric_cols].replace([np.inf, -np.inf], 0)
    
    # 2.2 去极值
    for col in numeric_cols:
        # 计算分位数
        Q1 = processed_data[col].quantile(0.01)
        Q3 = processed_data[col].quantile(0.99)
        IQR = Q3 - Q1
        
        # 限制在[Q1 - 1.5*IQR, Q3 + 1.5*IQR]范围内
        processed_data[col] = processed_data[col].clip(
            lower=Q1 - 1.5*IQR,
            upper=Q3 + 1.5*IQR
        )
    
    # 3. 标准化原始特征
    if is_train:
        scaler = StandardScaler()
        processed_data[feature_cols] = scaler.fit_transform(processed_data[feature_cols])
    else:
        if scaler is None:
            raise ValueError("测试集处理需要提供训练集的scaler")
        processed_data[feature_cols] = scaler.transform(processed_data[feature_cols])
    
    # 4. 选择重要特征
    if is_train:
        important_features = get_important_features(data, feature_cols, target_cols)
    else:
        important_features = IMPORTANT_FEATURES
    
    # 5. 创建衍生特征
    for col in important_features:
        # 滚动特征
        for window in [5, 10, 20, 30]:
            processed_data[f'{col}_mean_{window}'] = processed_data[col].rolling(window).mean()
            processed_data[f'{col}_std_{window}'] = processed_data[col].rolling(window).std()
        
        # 差分特征
        processed_data[f'{col}_diff1'] = processed_data[col].diff()
        processed_data[f'{col}_diff2'] = processed_data[col].diff().diff()
        processed_data[f'{col}_pct_change'] = processed_data[col].pct_change()
    
    # 6. 处理衍生特征的缺失值
    processed_data = processed_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 7. 再次检查无穷值
    inf_mask = np.isinf(processed_data)
    if inf_mask.any().any():
        logger.warning("发现衍生特征中的无穷值，将替换为0")
        processed_data = processed_data.replace([np.inf, -np.inf], 0)
    
    return processed_data, scaler

def get_important_features(data: pd.DataFrame, feature_cols: List[str], 
                         target_cols: List[str], n_features: int = 20) -> List[str]:
    """使用LightGBM获取重要特征
    
    Args:
        data: 输入数据
        feature_cols: 特征列名列表
        target_cols: 目标列名列表
        n_features: 需要返回的重要特征数量
    
    Returns:
        important_features: 重要特征列表
    """
    global IMPORTANT_FEATURES
    
    # 计算特征重要性
    importance_dict = {col: 0 for col in feature_cols}
    
    # 对每个目标变量训练一个简单的LightGBM
    for target in target_cols:
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )
        model.fit(data[feature_cols], data[target])
        
        # 累加每个目标的特征重要性
        for feat, imp in zip(feature_cols, model.feature_importances_):
            importance_dict[feat] += imp
    
    # 选择最重要的特征
    important_features = sorted(importance_dict.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:n_features]
    
    # 打印特征重要性
    logger.info("\n重要特征及其重要性分数:")
    for feat, score in important_features:
        logger.info(f"{feat}: {score:.4f}")
    
    IMPORTANT_FEATURES = [feat[0] for feat in important_features]
    return IMPORTANT_FEATURES

def create_interaction_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """创建特征交互
    
    Args:
        df: 输入数据框
        is_train: 是否为训练集
    
    Returns:
        interaction_features: 交互特征数据框
    """
    features = []
    feature_names = []
    
    # 使用重要特征
    base_cols = get_important_features(df, is_train=is_train)
    
    # 两特征交互
    for i in range(len(base_cols)):
        for j in range(i + 1, len(base_cols)):
            col1, col2 = base_cols[i], base_cols[j]
            
            # 加法交互
            sum_feature = df[col1] + df[col2]
            features.append(sum_feature)
            feature_names.append(f'{col1}_{col2}_sum')
            
            # 乘法交互
            mul_feature = df[col1] * df[col2]
            features.append(mul_feature)
            feature_names.append(f'{col1}_{col2}_mul')
            
            # 比交互
            ratio_feature = df[col1] / (df[col2] + 1e-8)
            features.append(ratio_feature)
            feature_names.append(f'{col1}_{col2}_ratio')
    
    # 创建数据框
    interaction_df = pd.DataFrame(np.column_stack(features), 
                                columns=feature_names,
                                index=df.index)
    
    return interaction_df

def create_statistical_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """创建统计特征
    
    Args:
        df: 输入数据框
        is_train: 是否为训练集
    
    Returns:
        statistical_features: 统计特征数据框
    """
    features = []
    feature_names = []
    
    # 使用重要特征
    base_cols = get_important_features(df, is_train=is_train)
    
    # 统计特征
    features.append(df[base_cols].mean(axis=1))
    feature_names.append('mean')
    
    features.append(df[base_cols].std(axis=1))
    feature_names.append('std')
    
    features.append(df[base_cols].max(axis=1))
    feature_names.append('max')
    
    features.append(df[base_cols].min(axis=1))
    feature_names.append('min')
    
    features.append(df[base_cols].max(axis=1) - df[base_cols].min(axis=1))
    feature_names.append('range')
    
    features.append(df[base_cols].skew(axis=1))
    feature_names.append('skew')
    
    features.append(df[base_cols].kurtosis(axis=1))
    feature_names.append('kurtosis')
    
    # 创建数据框
    statistical_df = pd.DataFrame(np.column_stack(features),
                                columns=feature_names,
                                index=df.index)
    
    return statistical_df

def create_time_series_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """创建时序特征
    
    Args:
        df: 输入数据框
        is_train: 是否为训练集
    
    Returns:
        time_series_features: 时序特征数据框
    """
    features = []
    feature_names = []
    
    # 使用重要特征
    base_cols = get_important_features(df, is_train=is_train)
    
    # 滑动窗口大小
    windows = [3, 5, 7]
    
    for col in base_cols:
        for window in windows:
            # 移动平均
            ma = df[col].rolling(window=window, min_periods=1).mean()
            features.append(ma)
            feature_names.append(f'{col}_ma_{window}')
            
            # 移动标准差
            std = df[col].rolling(window=window, min_periods=1).std()
            features.append(std)
            feature_names.append(f'{col}_std_{window}')
            
            # 移动最大值
            max_val = df[col].rolling(window=window, min_periods=1).max()
            features.append(max_val)
            feature_names.append(f'{col}_max_{window}')
            
            # 移动最小值
            min_val = df[col].rolling(window=window, min_periods=1).min()
            features.append(min_val)
            feature_names.append(f'{col}_min_{window}')
    
    # 创建数据框
    time_series_df = pd.DataFrame(np.column_stack(features),
                                columns=feature_names,
                                index=df.index)
    
    return time_series_df

def validate_data(data: pd.DataFrame, is_train: bool = True) -> None:
    """验证数据有效性"""
    # 检查索引
    assert data.index.name == 'ID', "索引必须为'ID'"
    
    # 检查特征列
    feature_cols = [col for col in data.columns if col.startswith('X')]
    assert len(feature_cols) > 0, "没有找到X特征列"
    
    # 如果是训练集，检查标签列
    if is_train:
        target_cols = [f'Y{i}' for i in range(8)]
        assert all(col in data.columns for col in target_cols), "训练集缺少Y标签列"
    
    # 检查数据类型
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(data[col]), f"{col}不是数值类型"
    
    # 检查缺失值
    null_counts = data[feature_cols].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"发现缺失值:\n{null_counts[null_counts > 0]}")
    
    # 检查无穷值
    inf_counts = np.isinf(data[feature_cols]).sum()
    if inf_counts.sum() > 0:
        logger.warning(f"发现无穷值:\n{inf_counts[inf_counts > 0]}")