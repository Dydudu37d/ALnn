# ALnn

一个基于Numpy的神经网络库，支持简单的前馈神经网络和卷积神经网络。

## 功能特性

- 支持多种激活函数（sigmoid、tanh、ReLU）
- 包含多逻辑门网络（AND、OR、XOR、NAND）
- 简单的卷积神经网络实现（MiniCNN）
- 支持CPU和GPU加速（通过Numba）
- 数据加载功能

## 安装

```bash
pip install -e .
```

## 使用示例

### 多逻辑门网络

```python
from ALnn import MultiLogicNet

# 初始化网络
model = MultiLogicNet()

# 训练数据
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0,0,0,1], [0,1,1,1], [0,1,1,1], [1,1,0,0]])

# 训练模型
model.train(X, y, epochs=10000)

# 评估模型
model.evaluate(X, y)
```

### 卷积神经网络

```python
from ALnn import MiniCNN, load_num_data

# 加载数据
X, y, imgs = load_num_data()

# 初始化CNN
cnn = MiniCNN()

# 训练模型
cnn.train(X[0], y[0], epochs=1000)

# 预测
result = cnn.predict(X[1])
```

## 许可证

MIT License