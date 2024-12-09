# 机器学习集成预测项目

## 项目说明
这是一个使用多个机器学习和深度学习模型进行集成预测的项目。项目集成了以下模型：
- 自编码器 (AutoEncoder)：用于特征降维
- CatBoost：梯度提升树模型
- GRU：门控循环单元网络
- Transformer：注意力机制模型
- Ridge：岭回归
- LightGBM：轻量级梯度提升树
- XGBoost：极限梯度提升树

## 项目结构
- config/: 配置文件
- data/: 数据文件
  - raw/: 原始数据
  - processed/: 处理后的数据
  - output/: 输出结果
- models/: 保存的模型
- notebooks/: Jupyter notebooks
- predictions/: 预测结果
- src/: 源代码
  - models/: 模型实现
  - utils/: 工具函数
- tests/: 测试代码
- logs/: 日志文件

## 安装依赖 