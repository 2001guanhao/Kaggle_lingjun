# 金融时序预测项目

## 项目设置

### 训练参数设置
1. 贝叶斯优化参数：
   - 最大尝试次数：10
   - 连续无改善次数：2（早停）
   - 优化时训练轮数：300
   - 最终训练轮数：3000
   - 模型训练早停轮数：100

2. 特征工程：
   - 重要特征选择：每个目标选择前20个特征
   - 滚动窗口大小：[5, 10, 20, 30]
   - 基本特征：
     * 滚动均值和标准差
     * 一阶和二阶差分
     * 百分比变化

3. 模型特定参数：
   - LightGBM：
     * feature_pre_filter: false
     * predict_disable_shape_check: true
     * 参数搜索范围：
       - num_leaves: [100, 300]
       - learning_rate: [0.005, 0.05]
       - feature_fraction: [0.6, 0.8]
       - bagging_fraction: [0.7, 0.9]
       - min_data_in_leaf: [500, 1500]
       - lambda_l1: [0.05, 0.2]
       - lambda_l2: [0.05, 0.2]

   - XGBoost：
     * verbosity: 1
     * 参数搜索范围：
       - max_depth: [3, 10]
       - eta: [0.001, 0.1]
       - subsample: [0.5, 1.0]
       - colsample_bytree: [0.5, 1.0]
       - min_child_weight: [1, 50]
       - lambda: [0, 10]
       - alpha: [0, 10]

   - CatBoost：
     * 参数搜索范围：
       - depth: [4, 10]
       - learning_rate: [0.001, 0.1]
       - l2_leaf_reg: [0, 10]
       - random_strength: [0, 10]
       - bagging_temperature: [0, 1]

4. 特征选择：
   - 使用第二个LightGBM进行特征选择
   - 合并所有目标变量的特征重要性
   - 选择前100个最重要特征用于深度学习模型

5. 训练流程：
   1. Ridge回归基础预测
   2. LightGBM（使用Ridge预测作为特征）
   3. XGBoost（使用Ridge预测作为特征）
   4. CatBoost（使用Ridge预测作为特征）
   5. 特征选择LightGBM（使用所有模型预测作为特征）
   6. 深度学习模型（使用选择的前100个特征）

## 使用说明

### 运行模式
1. 测试模式：
```bash
python src/main.py --test --samples 100
```

2. 仅机器学习模型：
```bash
python src/main.py --ml-only
```

3. 完整运行：
```bash
python src/main.py
```

### 环境要求
见 requirements.txt

## 项目结构
```
kaggle/lj/
├── config/              # 配置文件
│   └── model_config.yaml
├── data/               # 数据目录
│   ├── raw/           # 原始数据
│   ├── processed/     # 处理后的数据
│   └── output/        # 输出结果
├── docs/              # 文档
│   ├── model.md       # 模型文档
│   └── features.md    # 特征文档
├── logs/              # 日志文件
├── models/            # 保存的模型
├── notebooks/         # Jupyter notebooks
│   ├── eda.ipynb     # 数据分析
│   └── feature_analysis.ipynb
├── predictions/       # 预测结果
├── scripts/          # 脚本
│   ├── train.sh     # 训练脚本
│   └── predict.sh   # 预测脚本
├── src/              # 源代码
│   ├── models/      # 模型实现
│   └── utils/       # 工具函数
└── tests/           # 测试代码
```

## 特征工程

### 1. 基础特征
- 统计特征
  - 滚动均值(5/10/20/30分钟)
  - 滚动标准差
  - 滚动偏度/峰度
  - 滚动最大/最小值
  - 滚动分位数

- 差分特征
  - 一阶差分
  - 二阶差分
  - 百分比变化

### 2. 特征选择
- 相关性筛选
  - 去除高��相关特征(阈值0.95)
  - 去除低方差特征
  
- 重要性筛选
  - LightGBM特征重要性
  - 互信息得分

### 3. 特征验证
- 特征重要性分析
- 特征分布分析
- 特征相关性分析

## 模型架构

### 1. 模型流程
```
原始特征 + 特征工程
    ↓
Ridge回归 → Ridge预测结果
    ↓
合并特征集 = 原始特征 + 特征工程 + Ridge预测
    ↓
┌─────────┼─────────┬─────────┐
↓         ↓         ↓         
LightGBM  XGBoost  CatBoost   
    ↓         ↓         ↓     
预测结果  预测结果  预测结果  
    ↓         ↓         ↓     
    所有特征和预测结果 → LightGBM2
                ↓
        选择Top 100特征
                ↓
        ┌───────┴───────┐
        ↓               ↓
       GRU        Transformer
        ↓               ↓
    预测结果      预测结果
        ↓               ↓
            集成学习
```

### 2. 模型优化

#### 贝叶斯优化

1. 初始参数
```yaml
# LightGBM基础参数
objective: regression           # 目标为回归
metric: rmse                   # 评估指标
num_leaves: 200                # 叶子数量
learning_rate: 0.01            # 学习率
feature_fraction: 0.7          # 特征采样率
bagging_fraction: 0.8          # 数据采样率
bagging_freq: 1                # 每k次迭代执行bagging
max_bin: 512                   # 最大分箱数
min_data_in_leaf: 1000         # 叶子节点最少样本数
lambda_l1: 0.1                 # L1正则化
lambda_l2: 0.1                 # L2正则化
```

2. 优化流程
```
使用初始参数训练300轮
    ↓
贝叶斯优化4次(每次训练300轮)
    ↓
验证集RMSE评估
    ↓
连续2次无提升则停止
    ↓
使用最优参数正式训练
    ↓
训练设置：
- 训练轮数：20000
- 早停：100轮无提升
- 学习率：cosine衰减(从最优lr开始)
```

3. 参数搜索范围
```yaml
# LightGBM参数范围
num_leaves: [150, 300]         # 在基础参数200附近搜索
learning_rate: [0.005, 0.05]   # 在基础参数0.01附近搜索
feature_fraction: [0.6, 0.8]   # 在基础参数0.7附近搜索
bagging_fraction: [0.7, 0.9]   # 在基础参数0.8附近搜索
min_data_in_leaf: [800, 1200]  # 在基础参数1000附近搜索
lambda_l1: [0.05, 0.2]        # 在基础参数0.1附近搜索
lambda_l2: [0.05, 0.2]        # 在基础参数0.1附近搜索
```

4. 优化策略
- 对每个目标变量(Y0-Y7)单独优化
- 使用验证集RMSE作为优化目标
- 保存验证集表现最好的模型和参数
- 使用最佳模型生成预测，无需重新训练

5. 训练流程
```
初始化(使用基础参数)
    ↓
随机采样2次
    ↓
贝叶斯优化4次
    ↓
验证集RMSE评估
    ↓
连续2次无提升则停止
    ↓
使用最佳模型预测
```

6. 模型保存
- 保存最佳参数：`models/best_params_{model}_{target}.json`
- 保存最佳模型：`models/best_model_{model}_{target}.txt`
- 保存优化历史：`models/opt_history_{model}_{target}.json`

### 3. 评估指标
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

## 安装和使用

### 1. 环境要求
```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
lightgbm>=3.3.0
xgboost>=1.5.0
catboost>=1.0.0
torch>=1.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
optuna>=2.10.0
mlflow>=1.20.0
```

### 2. 安装步骤
```bash
# 克隆仓库
git clone https://github.com/2001guanhao/Kaggle_lingjun.git
cd Kaggle_lingjun

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 使用说明
```bash
# 完整训练
python src/main.py

# 测试模式
python src/main.py --test --samples 100

# 仅机器学习模型
python src/main.py --ml-only
```

## 模型保存
- Ridge: 不保存（简单线性模型）
- LightGBM: 每个目标变量的模型单独保存（.txt格式）
- XGBoost: 每个目标变量的模型单独保存（.json格式）
- GRU: 保存最佳checkpoint（.pt格式）
- Transformer: 保存最佳checkpoint（.pt格式）

保存路径：`models/`目录
文件命名格式：`{model_name}_{target_col}_{timestamp}.{ext}`

## 开发计划
- [ ] 添加交叉验证
- [ ] 实现模型可解释分析
- [ ] 添加特征工程pipeline
- [ ] 优化模型集成策略
- [ ] 添加内存使用监控
- [ ] 完善错误处理机制
- [ ] 添加模型训练检查点
- [ ] 实现并行训练
- [ ] 添加实验跟踪

## 许可证
本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 作者
- 关昊 (@2001guanhao)