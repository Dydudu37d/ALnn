import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
from colorama import Fore, Back, Style, init

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查是否有可用的GPU
has_gpu = False
try:
    from numba import cuda
    if len(cuda.gpus) > 0:
        has_gpu = True
        print("检测到可用GPU，将使用CUDA加速")
    else:
        print("未检测到可用GPU，将使用CPU")
except ImportError:
    print("CUDA不可用，将使用CPU")

# 条件导入numba
try:
    from numba import jit, vectorize
    print("Numba已加载")
except ImportError:
    print("Numba不可用")
    jit = lambda *args, **kwargs: lambda f: f  # 定义一个空装饰器
    vectorize = lambda *args, **kwargs: lambda f: f

# 激活函数及其导数 - 条件使用numba优化
if has_gpu:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    def mean_squared_error(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def tanh(x):
        return np.tanh(x)
    
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
    
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return 1.0 * (x > 0)
else:
    @jit(nopython=True, parallel=True)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @jit(nopython=True, parallel=True)
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    @jit(nopython=True, parallel=True)
    def mean_squared_error(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    @jit(nopython=True, parallel=True)
    def tanh(x):
        return np.tanh(x)
    
    @jit(nopython=True, parallel=True)
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
    
    @jit(nopython=True, parallel=True)
    def relu(x):
        return np.maximum(0, x)
    
    @jit(nopython=True, parallel=True)
    def relu_derivative(x):
        # 对于向量输入，逐个元素处理
        if isinstance(x, np.ndarray):
            result = np.zeros_like(x)
            for i in range(x.size):
                if x.flat[i] > 0:
                    result.flat[i] = 1
                else:
                    result.flat[i] = 0
            return result
        # 对于标量输入
        else:
            return 1 if x > 0 else 0

# 数据准备 - 四种逻辑门的真实输出
X = []
y = []
img_list = []

# 核心CUDA优化函数 - 条件实现
def create_forward_tanh():
    if has_gpu:
        # CUDA内核函数（不是设备函数）
        @cuda.jit
        def forward_tanh_kernel(X, W1, b1, W2, b2, a1, a2):
            """前向传播CUDA内核 - tanh激活"""
            row = cuda.grid(1)
            if row < X.shape[0]:
                # 计算第一个隐藏层的每个神经元
                for neuron_idx in range(W1.shape[1]):
                    a1_val = 0.0
                    for k in range(X.shape[1]):
                        a1_val += X[row, k] * W1[k, neuron_idx]
                    a1_val += b1[0, neuron_idx]
                    a1_val = math.tanh(a1_val)
                    a1[row, neuron_idx] = a1_val
            
            # 同步以确保a1计算完成
            cuda.syncthreads()
            
            # 计算输出层
            if row < a1.shape[0]:
                for out_col in range(a2.shape[1]):
                    a2_val = 0.0
                    for k in range(a1.shape[1]):
                        a2_val += a1[row, k] * W2[k, out_col]
                    a2_val += b2[0, out_col]
                    a2[row, out_col] = 1.0 / (1.0 + math.exp(-a2_val))  # sigmoid
        
        # 主机包装函数 - 管理内存传输
        def forward_tanh_host(X, W1, b1, W2, b2):
            """前向传播主机包装函数 - tanh激活"""
            import math
            # 确保输入是float64类型
            X_float = X.astype(np.float64)
            
            # 创建输出数组
            a1 = np.zeros((X_float.shape[0], W1.shape[1]), dtype=np.float64)
            a2 = np.zeros((X_float.shape[0], W2.shape[1]), dtype=np.float64)
            
            # 传输数据到GPU
            X_device = cuda.to_device(X_float)
            W1_device = cuda.to_device(W1)
            b1_device = cuda.to_device(b1)
            W2_device = cuda.to_device(W2)
            b2_device = cuda.to_device(b2)
            a1_device = cuda.device_array_like(a1)
            a2_device = cuda.device_array_like(a2)
            
            # 设置CUDA网格和块大小 - 使用较小的块大小
            threads_per_block = 128
            blocks_per_grid = (X_float.shape[0] + threads_per_block - 1) // threads_per_block
            
            # 启动内核
            forward_tanh_kernel[blocks_per_grid, threads_per_block](
                X_device, W1_device, b1_device, W2_device, b2_device, a1_device, a2_device
            )
            
            # 从GPU获取结果
            a1 = a1_device.copy_to_host()
            a2 = a2_device.copy_to_host()
            
            return a1, a2
        
        return forward_tanh_host
    else:
        # CPU版本
        @jit(nopython=True, parallel=True)
        def forward_tanh_cpu(X, W1, b1, W2, b2):
            """前向传播 - tanh激活（CPU优化）"""
            # 转换X为float64类型以匹配权重的数据类型
            X_float = X.astype(np.float64)
            # 第一层：输入 -> 隐藏层
            z1 = np.dot(X_float, W1)
            # 确保b1是一维数组并进行广播加法
            b1_reshaped = b1.reshape(1, -1)
            z1 += b1_reshaped
            a1 = np.tanh(z1)
            
            # 第二层：隐藏层 -> 输出层
            z2 = np.dot(a1, W2)
            # 确保b2是一维数组并进行广播加法
            b2_reshaped = b2.reshape(1, -1)
            z2 += b2_reshaped
            a2 = 1 / (1 + np.exp(-z2))  # sigmoid函数
            return a1, a2
        
        return forward_tanh_cpu

# 创建前向传播函数
forward_tanh = create_forward_tanh()

# 创建前向传播sigmoid函数（类似实现）
def create_forward_sigmoid():
    if has_gpu:
        # CUDA内核函数
        @cuda.jit
        def forward_sigmoid_kernel(X, W1, b1, W2, b2, a1, a2):
            """前向传播CUDA内核 - sigmoid激活"""
            row = cuda.grid(1)
            if row < X.shape[0]:
                # 计算第一个隐藏层的每个神经元
                for neuron_idx in range(W1.shape[1]):
                    a1_val = 0.0
                    for k in range(X.shape[1]):
                        a1_val += X[row, k] * W1[k, neuron_idx]
                    a1_val += b1[0, neuron_idx]
                    a1_val = 1.0 / (1.0 + math.exp(-a1_val))  # sigmoid
                    a1[row, neuron_idx] = a1_val
            
            # 同步以确保a1计算完成
            cuda.syncthreads()
            
            # 计算输出层
            if row < a1.shape[0]:
                for out_col in range(a2.shape[1]):
                    a2_val = 0.0
                    for k in range(a1.shape[1]):
                        a2_val += a1[row, k] * W2[k, out_col]
                    a2_val += b2[0, out_col]
                    a2[row, out_col] = 1.0 / (1.0 + math.exp(-a2_val))  # sigmoid
        
        # 主机包装函数
        def forward_sigmoid_host(X, W1, b1, W2, b2):
            """前向传播主机包装函数 - sigmoid激活"""
            import math
            # 确保输入是float64类型
            X_float = X.astype(np.float64)
            
            # 创建输出数组
            a1 = np.zeros((X_float.shape[0], W1.shape[1]), dtype=np.float64)
            a2 = np.zeros((X_float.shape[0], W2.shape[1]), dtype=np.float64)
            
            # 传输数据到GPU
            X_device = cuda.to_device(X_float)
            W1_device = cuda.to_device(W1)
            b1_device = cuda.to_device(b1)
            W2_device = cuda.to_device(W2)
            b2_device = cuda.to_device(b2)
            a1_device = cuda.device_array_like(a1)
            a2_device = cuda.device_array_like(a2)
            
            # 设置CUDA网格和块大小
            threads_per_block = 128
            blocks_per_grid = (X_float.shape[0] + threads_per_block - 1) // threads_per_block
            
            # 启动内核
            forward_sigmoid_kernel[blocks_per_grid, threads_per_block](
                X_device, W1_device, b1_device, W2_device, b2_device, a1_device, a2_device
            )
            
            # 从GPU获取结果
            a1 = a1_device.copy_to_host()
            a2 = a2_device.copy_to_host()
            
            return a1, a2
        
        return forward_sigmoid_host
    else:
        # CPU版本
        @jit(nopython=True, parallel=True)
        def forward_sigmoid_cpu(X, W1, b1, W2, b2):
            """前向传播 - sigmoid激活（CPU优化）"""
            # 转换X为float64类型以匹配权重的数据类型
            X_float = X.astype(np.float64)
            # 第一层：输入 -> 隐藏层
            z1 = np.dot(X_float, W1)
            # 确保b1是一维数组并进行广播加法
            b1_reshaped = b1.reshape(1, -1)
            z1 += b1_reshaped
            a1 = 1 / (1 + np.exp(-z1))  # sigmoid函数
            
            # 第二层：隐藏层 -> 输出层
            z2 = np.dot(a1, W2)
            # 确保b2是一维数组并进行广播加法
            b2_reshaped = b2.reshape(1, -1)
            z2 += b2_reshaped
            a2 = 1 / (1 + np.exp(-z2))  # sigmoid函数
            return a1, a2
        
        return forward_sigmoid_cpu

# 创建前向传播sigmoid函数
forward_sigmoid = create_forward_sigmoid()

# 创建反向传播tanh函数
def create_backward_tanh():
    if has_gpu:
        # CUDA内核函数
        @cuda.jit
        def backward_tanh_kernel(X, y, a1, a2, W2, dW1, db1, dW2, db2):
            """反向传播CUDA内核 - tanh激活"""
            row = cuda.grid(1)
            # 计算输出层梯度
            if row < a2.shape[0]:
                for out_col in range(a2.shape[1]):
                    dZ2 = a2[row, out_col] - y[row, out_col]
                    # 计算dW2
                    for h_col in range(a1.shape[1]):
                        cuda.atomic.add(dW2, (h_col, out_col), a1[row, h_col] * dZ2)
                    # 计算db2
                    cuda.atomic.add(db2, (0, out_col), dZ2)
            
            # 同步
            cuda.syncthreads()
            
            # 计算隐藏层梯度
            if row < a1.shape[0]:
                for h_col in range(a1.shape[1]):
                    dZ1 = 0.0
                    for out_col in range(W2.shape[1]):
                        dZ1 += (a2[row, out_col] - y[row, out_col]) * W2[h_col, out_col]
                    dZ1 *= (1 - a1[row, h_col] ** 2)  # tanh导数
                    # 计算dW1
                    for in_col in range(X.shape[1]):
                        cuda.atomic.add(dW1, (in_col, h_col), X[row, in_col] * dZ1)
                    # 计算db1
                    cuda.atomic.add(db1, (0, h_col), dZ1)
        
        # 主机包装函数
        def backward_tanh_host(X, y, a1, a2, W2, m):
            """反向传播主机包装函数 - tanh激活"""
            # 确保输入是float64类型
            X_float = X.astype(np.float64)
            
            # 创建梯度数组
            dW1 = np.zeros((X_float.shape[1], a1.shape[1]), dtype=np.float64)
            db1 = np.zeros((1, a1.shape[1]), dtype=np.float64)
            dW2 = np.zeros((a1.shape[1], a2.shape[1]), dtype=np.float64)
            db2 = np.zeros((1, a2.shape[1]), dtype=np.float64)
            
            # 传输数据到GPU
            X_device = cuda.to_device(X_float)
            y_device = cuda.to_device(y)
            a1_device = cuda.to_device(a1)
            a2_device = cuda.to_device(a2)
            W2_device = cuda.to_device(W2)
            dW1_device = cuda.device_array_like(dW1)
            db1_device = cuda.device_array_like(db1)
            dW2_device = cuda.device_array_like(dW2)
            db2_device = cuda.device_array_like(db2)
            
            # 设置CUDA网格和块大小
            threads_per_block = 128
            blocks_per_grid = (X_float.shape[0] + threads_per_block - 1) // threads_per_block
            
            # 启动内核
            backward_tanh_kernel[blocks_per_grid, threads_per_block](
                X_device, y_device, a1_device, a2_device, W2_device,
                dW1_device, db1_device, dW2_device, db2_device
            )
            
            # 从GPU获取结果
            dW1 = dW1_device.copy_to_host() / m
            db1 = db1_device.copy_to_host() / m
            dW2 = dW2_device.copy_to_host() / m
            db2 = db2_device.copy_to_host() / m
            
            return dW1, db1, dW2, db2
        
        return backward_tanh_host
    else:
        # CPU版本
        @jit(nopython=True, parallel=True)
        def backward_tanh_cpu(X, y, a1, a2, W2, m):
            """反向传播 - tanh激活（CPU优化）"""
            # 转换X为float64类型以匹配权重的数据类型
            X_float = X.astype(np.float64)
            # 输出层误差和梯度
            dZ2 = a2 - y
            dW2 = (1 / m) * np.dot(a1.T, dZ2)
            db2 = (1 / m) * np.sum(dZ2, axis=0)
            # 手动重塑db2为(1, n)形状
            db2 = db2.reshape(1, -1)
            
            # 隐藏层误差和梯度
            dZ1 = np.dot(dZ2, W2.T) * (1 - a1 ** 2)  # tanh导数
            dW1 = (1 / m) * np.dot(X_float.T, dZ1)
            db1 = (1 / m) * np.sum(dZ1, axis=0)
            # 手动重塑db1为(1, n)形状
            db1 = db1.reshape(1, -1)
            
            return dW1, db1, dW2, db2
        
        return backward_tanh_cpu

# 创建反向传播tanh函数
backward_tanh = create_backward_tanh()

# 创建反向传播sigmoid函数
def create_backward_sigmoid():
    if has_gpu:
        # CUDA内核函数
        @cuda.jit
        def backward_sigmoid_kernel(X, y, a1, a2, W2, dW1, db1, dW2, db2):
            """反向传播CUDA内核 - sigmoid激活"""
            row = cuda.grid(1)
            # 计算输出层梯度
            if row < a2.shape[0]:
                for out_col in range(a2.shape[1]):
                    dZ2 = a2[row, out_col] - y[row, out_col]
                    # 计算dW2
                    for h_col in range(a1.shape[1]):
                        cuda.atomic.add(dW2, (h_col, out_col), a1[row, h_col] * dZ2)
                    # 计算db2
                    cuda.atomic.add(db2, (0, out_col), dZ2)
            
            # 同步
            cuda.syncthreads()
            
            # 计算隐藏层梯度
            if row < a1.shape[0]:
                for h_col in range(a1.shape[1]):
                    dZ1 = 0.0
                    for out_col in range(W2.shape[1]):
                        dZ1 += (a2[row, out_col] - y[row, out_col]) * W2[h_col, out_col]
                    dZ1 *= a1[row, h_col] * (1 - a1[row, h_col])  # sigmoid导数
                    # 计算dW1
                    for in_col in range(X.shape[1]):
                        cuda.atomic.add(dW1, (in_col, h_col), X[row, in_col] * dZ1)
                    # 计算db1
                    cuda.atomic.add(db1, (0, h_col), dZ1)
        
        # 主机包装函数
        def backward_sigmoid_host(X, y, a1, a2, W2, m):
            """反向传播主机包装函数 - sigmoid激活"""
            # 确保输入是float64类型
            X_float = X.astype(np.float64)
            
            # 创建梯度数组
            dW1 = np.zeros((X_float.shape[1], a1.shape[1]), dtype=np.float64)
            db1 = np.zeros((1, a1.shape[1]), dtype=np.float64)
            dW2 = np.zeros((a1.shape[1], a2.shape[1]), dtype=np.float64)
            db2 = np.zeros((1, a2.shape[1]), dtype=np.float64)
            
            # 传输数据到GPU
            X_device = cuda.to_device(X_float)
            y_device = cuda.to_device(y)
            a1_device = cuda.to_device(a1)
            a2_device = cuda.to_device(a2)
            W2_device = cuda.to_device(W2)
            dW1_device = cuda.device_array_like(dW1)
            db1_device = cuda.device_array_like(db1)
            dW2_device = cuda.device_array_like(dW2)
            db2_device = cuda.device_array_like(db2)
            
            # 设置CUDA网格和块大小
            threads_per_block = 128
            blocks_per_grid = (X_float.shape[0] + threads_per_block - 1) // threads_per_block
            
            # 启动内核
            backward_sigmoid_kernel[blocks_per_grid, threads_per_block](
                X_device, y_device, a1_device, a2_device, W2_device,
                dW1_device, db1_device, dW2_device, db2_device
            )
            
            # 从GPU获取结果
            dW1 = dW1_device.copy_to_host() / m
            db1 = db1_device.copy_to_host() / m
            dW2 = dW2_device.copy_to_host() / m
            db2 = db2_device.copy_to_host() / m
            
            return dW1, db1, dW2, db2
        
        return backward_sigmoid_host
    else:
        # CPU版本
        @jit(nopython=True, parallel=True)
        def backward_sigmoid_cpu(X, y, a1, a2, W2, m):
            """反向传播 - sigmoid激活（CPU优化）"""
            # 转换X为float64类型以匹配权重的数据类型
            X_float = X.astype(np.float64)
            # 输出层误差和梯度
            dZ2 = a2 - y
            dW2 = (1 / m) * np.dot(a1.T, dZ2)
            db2 = (1 / m) * np.sum(dZ2, axis=0)
            # 手动重塑db2为(1, n)形状
            db2 = db2.reshape(1, -1)
            
            # 隐藏层误差和梯度
            dZ1 = np.dot(dZ2, W2.T) * a1 * (1 - a1)  # sigmoid导数
            dW1 = (1 / m) * np.dot(X_float.T, dZ1)
            db1 = (1 / m) * np.sum(dZ1, axis=0)
            # 手动重塑db1为(1, n)形状
            db1 = db1.reshape(1, -1)
            
            return dW1, db1, dW2, db2
        
        return backward_sigmoid_cpu

# 创建反向传播sigmoid函数
backward_sigmoid = create_backward_sigmoid()

class MultiLogicNet:
    def __init__(self, input_size=2, hidden_size=8, output_size=4, activation='tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.activation = activation
        
        # 权重初始化
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        self.gate_names = ['AND', 'OR', 'XOR', 'NAND']
    
    def forward(self, X):
        """前向传播"""
        if self.activation == 'tanh':
            return forward_tanh(X, self.W1, self.b1, self.W2, self.b2)
        else:
            return forward_sigmoid(X, self.W1, self.b1, self.W2, self.b2)
    
    def backward(self, X, y, a1, a2):
        """反向传播"""
        m = X.shape[0]  # 样本数量
        
        if self.activation == 'tanh':
            return backward_tanh(X, y, a1, a2, self.W2, m)
        else:
            return backward_sigmoid(X, y, a1, a2, self.W2, m)
    
    def train(self, X, y, learning_rate=0.1, epochs=10000):
        print(f"开始训练多逻辑门网络...")
        print(f"网络结构: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
        
        for epoch in range(epochs):
            # 前向传播
            a1, a2 = self.forward(X)
            
            # 反向传播
            dW1, db1, dW2, db2 = self.backward(X, y, a1, a2)
            
            # 更新权重和偏置
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            
            # 每1000次迭代打印一次损失
            if epoch % 1000 == 0:
                loss = mean_squared_error(a2, y)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict_all_gates(self, X):
        """同时预测4种逻辑门的结果"""
        _, predictions = self.forward(X)
        binary_predictions = (predictions > 0.5).astype(int)
        return predictions, binary_predictions
    
    def evaluate(self, X, y):
        """评估模型在所有逻辑门上的表现"""
        predictions, binary_predictions = self.predict_all_gates(X)
        
        print("\n" + "="*50)
        print("多逻辑门网络评估结果")
        print("="*50)
        
        for i in range(len(X)):
            print(f"\n输入: {X[i]}")
            for gate_idx, gate_name in enumerate(self.gate_names):
                pred_prob = predictions[i][gate_idx]
                pred_binary = binary_predictions[i][gate_idx] 
                true_val = y[i][gate_idx]
                status = "✓" if pred_binary == true_val else "✗"
                print(f"  {gate_name}: {pred_prob:.4f} -> {pred_binary} {status} (真实: {true_val})")
    
    def predict(self, X):
        """單個樣本預測"""
        _, predictions = self.forward(X)
        binary_predictions = (predictions > 0.5).astype(int)
        return predictions, binary_predictions
    
    def save(self, filename='multi_logic_net.npz'):
        """保存模型權重"""
        np.savez(filename, W1=self.W1, b1=self.b1,W2=self.W2, b2=self.b2)
        print(f"模型已保存至 {filename}")
    
    def load(self, filename='multi_logic_net.npz'):
        """載入模型權重"""
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"模型已載入自 {filename}")

def visualize_cnn_workflow():
        """可視化CNN的每一步處理"""
        # 創建一個簡單的測試圖像（5x5）
        test_image = np.array([
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0], 
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1]
        ])
        
        # 創建邊緣檢測卷積核
        edge_kernel = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ])
        
        print("原始圖像:")
        print(test_image)
        
        # 卷積操作
        cnn = SimpleCNN()
        conv_result = cnn.conv2d(test_image, edge_kernel)
        print("\n卷積結果:")
        print(conv_result)
        
        # ReLU激活
        relu_result = cnn.relu(conv_result)
        print("\nReLU激活後:")
        print(relu_result)
        
        # 池化
        pool_result = cnn.max_pooling(relu_result)
        print("\n池化後:")
        print(pool_result)
        
        return test_image, conv_result, relu_result, pool_result

# MiniCNN优化函数 - 条件式CUDA优化
# 创建CNN卷积函数
def create_cnn_conv2d():
    if has_gpu:
        # CUDA内核函数 - 卷积操作
        @cuda.jit
        def cnn_conv2d_kernel(input_img, kernel, output):
            """卷积操作CUDA内核"""
            # 获取当前线程的全局索引
            i, j = cuda.grid(2)
            
            if i < output.shape[0] and j < output.shape[1]:
                # 计算卷积
                conv_sum = 0.0
                for ki in range(kernel.shape[0]):
                    for kj in range(kernel.shape[1]):
                        if 0 <= i + ki < input_img.shape[0] and 0 <= j + kj < input_img.shape[1]:
                            conv_sum += input_img[i + ki, j + kj] * kernel[ki, kj]
                output[i, j] = conv_sum
        
        # 主机包装函数
        def cnn_conv2d_host(input_img, kernel):
            """卷积操作主机包装函数"""
            # 确保输入是float64类型
            input_float = input_img.astype(np.float64)
            kernel_float = kernel.astype(np.float64)
            
            # 计算输出形状
            output_h = input_float.shape[0] - kernel_float.shape[0] + 1
            output_w = input_float.shape[1] - kernel_float.shape[1] + 1
            output = np.zeros((output_h, output_w), dtype=np.float64)
            
            # 传输数据到GPU
            input_device = cuda.to_device(input_float)
            kernel_device = cuda.to_device(kernel_float)
            output_device = cuda.device_array_like(output)
            
            # 设置CUDA网格和块大小
            threads_per_block = (16, 16)
            blocks_per_grid_x = (output.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (output.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            # 启动内核
            cnn_conv2d_kernel[blocks_per_grid, threads_per_block](
                input_device, kernel_device, output_device
            )
            
            # 从GPU获取结果
            output = output_device.copy_to_host()
            return output
        
        return cnn_conv2d_host
    else:
        # CPU版本
        @jit(nopython=True, parallel=True)
        def cnn_conv2d_cpu(input_img, kernel):
            """卷积操作（CPU优化）"""
            input_h, input_w = input_img.shape
            kernel_h, kernel_w = kernel.shape
            output_h = input_h - kernel_h + 1
            output_w = input_w - kernel_w + 1
            output = np.zeros((output_h, output_w), dtype=np.float64)
            
            for i in range(output_h):
                for j in range(output_w):
                    region = input_img[i:i+kernel_h, j:j+kernel_w]
                    output[i, j] = np.sum(region * kernel)
            return output
        
        return cnn_conv2d_cpu

# 创建卷积函数
cnn_conv2d = create_cnn_conv2d()

# 创建CNN最大池化函数
def create_cnn_max_pooling():
    if has_gpu:
        # CUDA内核函数 - 最大池化操作
        @cuda.jit
        def cnn_max_pooling_kernel(input_img, pool_size, output):
            """最大池化CUDA内核"""
            # 获取当前线程的全局索引
            i, j = cuda.grid(2)
            
            if i < output.shape[0] and j < output.shape[1]:
                # 计算池化区域的最大值
                max_val = -1e308  # 使用一个非常小的负数代替float('-inf')
                for pi in range(pool_size):
                    for pj in range(pool_size):
                        input_i = i * pool_size + pi
                        input_j = j * pool_size + pj
                        if 0 <= input_i < input_img.shape[0] and 0 <= input_j < input_img.shape[1]:
                            if input_img[input_i, input_j] > max_val:
                                max_val = input_img[input_i, input_j]
                output[i, j] = max_val
        
        # 主机包装函数
        def cnn_max_pooling_host(input_img, pool_size=2):
            """最大池化主机包装函数"""
            # 确保输入是float64类型
            input_float = input_img.astype(np.float64)
            
            # 计算输出形状
            output_h = input_float.shape[0] // pool_size
            output_w = input_float.shape[1] // pool_size
            output = np.zeros((output_h, output_w), dtype=np.float64)
            
            # 传输数据到GPU
            input_device = cuda.to_device(input_float)
            output_device = cuda.device_array_like(output)
            
            # 设置CUDA网格和块大小
            threads_per_block = (16, 16)
            blocks_per_grid_x = (output.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (output.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            # 启动内核
            cnn_max_pooling_kernel[blocks_per_grid, threads_per_block](
                input_device, pool_size, output_device
            )
            
            # 从GPU获取结果
            output = output_device.copy_to_host()
            return output
        
        return cnn_max_pooling_host
    else:
        # CPU版本
        @jit(nopython=True, parallel=True)
        def cnn_max_pooling_cpu(input_img, pool_size=2):
            """最大池化操作（CPU优化）"""
            h, w = input_img.shape
            output_h = h // pool_size
            output_w = w // pool_size
            output = np.zeros((output_h, output_w), dtype=np.float64)
            
            for i in range(output_h):
                for j in range(output_w):
                    region = input_img[i*pool_size:(i+1)*pool_size, 
                                     j*pool_size:(j+1)*pool_size]
                    output[i, j] = np.max(region)
            return output
        
        return cnn_max_pooling_cpu

# 创建最大池化函数
cnn_max_pooling = create_cnn_max_pooling()

# 移除device=True装饰器的cnn_softmax函数
def cnn_softmax(x):
    """softmax激活函数"""
    # 数值稳定性：减去最大值
    max_val = np.max(x)
    exp_x = np.exp(x - max_val)
    return exp_x / np.sum(exp_x)

# 移除device=True装饰器的backprop_core函数
def backprop_core(flattened, probabilities, y):
    """反向传播核心计算"""
    # 计算输出误差
    output_error = probabilities - y
    
    # 计算权重梯度
    d_weights = np.outer(flattened, output_error)
    d_bias = output_error
    
    return d_weights, d_bias

class MiniCNN:
    """迷你CNN，用於手寫數字識別"""
    
    def __init__(self, input_shape=(28, 28), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # 初始化卷積核（簡單的邊緣檢測器）
        self.conv_filters = [
            np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),  # 水平邊緣
            np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),  # 垂直邊緣
        ]
        
        # 全連接層權重（適用於28x28輸入圖像）
        # 28x28輸入 -> 3x3卷積 -> 26x26特徵圖 -> 2x2池化 -> 13x13特徵圖
        # 2個卷積核，所以特徵數為13*13*2=338
        self.weights = np.random.randn(13*13*2, num_classes) * 0.1
        self.bias = np.zeros(num_classes)
    
    def conv2d(self, input_img, kernel):
        """卷积操作（使用numba优化）"""
        return cnn_conv2d(input_img, kernel)
    
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def max_pooling(self, input_img, pool_size=2):
        """最大池化操作（使用numba优化）"""
        return cnn_max_pooling(input_img, pool_size)
    
    def softmax(self, x):
        """softmax激活函数（使用numba优化）"""
        # 使用优化的softmax实现
        exp_x = np.exp(x - np.max(x))  # 数值稳定性
        return exp_x / np.sum(exp_x)
    
    def forward(self, X):
        """前向傳播"""
        # 确保X是numpy数组
        X = np.array(X)
        # 确保输入是二维数组
        if len(X.shape) != 2:
            X = X.reshape(int(np.sqrt(X.size)), -1)
        
        # 卷积层
        conv_results = []
        for kernel in self.conv_filters:
            conv_result = self.conv2d(X, kernel)
            conv_results.append(conv_result)
        
        # ReLU激活
        relu_results = [self.relu(conv) for conv in conv_results]
        
        # 池化层
        pool_results = [self.max_pooling(relu) for relu in relu_results]
        
        # 展平
        flattened = np.concatenate([pool.flatten() for pool in pool_results])
        
        # 检查并调整权重矩阵大小
        if flattened.shape[0] != self.weights.shape[0]:
            print(f"前向传播中调整权重矩阵大小: {self.weights.shape} -> ({flattened.shape[0]}, 10)")
            self.weights = np.random.randn(flattened.shape[0], 10) * 0.1
        
        # 全连接层
        output = np.dot(flattened, self.weights) + self.bias
        probabilities = self.softmax(output)
        
        return probabilities, pool_results
    
    def backprop(self, X, y, probabilities, pool_results, learning_rate=0.01):
        """反向傳播 - 简化为单个样本处理"""
        # 确保X和y是numpy数组
        X = np.array(X)
        y = np.array(y)
        
        # 检查并调整y的形状，确保它是一维向量
        if len(y.shape) > 1:
            # 如果y是二维数组，取第一个样本
            y = y[0]
        
        # 确保probabilities的形状正确
        if len(probabilities.shape) > 1 and probabilities.shape[1] == 10:
            # 如果probabilities是二维数组，取第一个样本
            probabilities = probabilities[0]
        
        # 确保output_error是正确的形状 (10,)
        if len(probabilities.shape) > 1:
            probabilities = probabilities.flatten()
        
        # 获取特征向量
        flattened = np.concatenate([pool.flatten() for pool in pool_results])
        
        # 使用优化的核心函数计算梯度
        d_weights, d_bias = backprop_core(flattened, probabilities, y)
        
        # 确保梯度形状匹配权重矩阵
        if d_weights.shape != self.weights.shape:
            print(f"警告：梯度形状 {d_weights.shape} 与权重形状 {self.weights.shape} 不匹配")
            # 调整梯度形状以匹配权重
            if d_weights.shape[0] == self.weights.shape[0] and d_weights.shape[1] > self.weights.shape[1]:
                # 如果行数相同但列数不同，取前10列
                d_weights = d_weights[:, :self.weights.shape[1]]
            elif d_weights.shape[1] == self.weights.shape[1] and d_weights.shape[0] > self.weights.shape[0]:
                # 如果列数相同但行数不同，取前N行
                d_weights = d_weights[:self.weights.shape[0], :]
        
        # 更新权重和偏置
        try:
            self.weights -= learning_rate * d_weights
            self.bias -= learning_rate * d_bias
        except ValueError as e:
            print(f"更新权重时出错: {e}")
            print(f"权重形状: {self.weights.shape}, 梯度形状: {d_weights.shape}")
            print(f"偏置形状: {self.bias.shape}, 偏置梯度形状: {d_bias.shape}")
    
    def train(self, X, y, learning_rate=0.01, epochs=1000):
        """訓練模型"""
        for epoch in range(epochs):
            # 前向傳播
            probabilities, pool_results = self.forward(X)
            
            # 反向傳播
            self.backprop(X, y, probabilities, pool_results, learning_rate)
            
            # 每50次迭代打印一次损失
            if epoch % 50 == 0:
                # 初始化colorama以确保在Windows上正确显示颜色
                init()
                loss = mean_squared_error(probabilities, y)
                # 计算进度百分比
                progress_percent = (epoch / epochs) * 101
                bar_length = 30
                filled_length = int(bar_length * epoch / epochs) + 1
                # 创建进度条
                bar = Fore.GREEN + '█' * filled_length + Fore.WHITE + '-' * (bar_length - filled_length) 
                # 只使用一个print语句，避免输出被覆盖
                print(f"{Fore.BLUE}Epoch {epoch:5d}/{epochs}, Loss: {loss:.4f} {Fore.RESET}[{bar}] {progress_percent:.0f}%", end='\r')
                # 如果是最后一个epoch，添加换行
                if epoch == epochs:
                    print()
        print()
    
    def predict(self, X):
        """預測模型"""
        probabilities, _ = self.forward(X)
        return probabilities
    
    def save(self, filename='mini_cnn.npz'):
        """保存模型權重"""
        np.savez(filename, conv_filters=self.conv_filters,weights=self.weights, bias=self.bias)
        print(f"模型已保存至 {filename}")
    
    def load(self, filename='mini_cnn.npz'):
        """載入模型權重"""
        data = np.load(filename)
        self.conv_filters = data['conv_filters']
        self.weights = data['weights']
        self.bias = data['bias']
        print(f"模型已載入自 {filename}")

def load_num_data(path=None, show=True):
    """加載數據集并处理28x28图像"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'num')
    
    X = []
    y = []
    img_list = []
    
    for i in os.listdir(path):
        if i.isdigit():
            digit_folder = os.path.join(path, i)
            if os.path.isdir(digit_folder):
                for j in os.listdir(digit_folder):
                    if j.endswith('.png'):
                        try:
                            img_path = os.path.join(digit_folder, j)
                            img = plt.imread(img_path)
                            if show:
                                plt.imshow(img, cmap='gray')
                                plt.title(f"number {i}")
                            
                            # 确保是灰度图
                            if len(img.shape) > 2:
                                img = img[:, :, 0]  # 取第一个通道
                            
                            # 检查图像大小
                            current_size = img.shape
                            print(f"加載數字 {i}，原始形狀: {current_size}")
                            
                            # 將圖像轉換為0-1範圍
                            img = img / 255.0 if img.max() > 1 else img
                            
                            # 展平圖像為1D向量
                            img_flat = img.flatten()
                            
                            # 將展平的向量添加到數據集中
                            X.append(img_flat)
                            # 添加對應的標籤
                            y.append(np.array([1 if int(i) == j else 0 for j in range(10)]))
                            img_list.append(img)
                            
                            if show:
                                plt.show()
                        except Exception as e:
                            print(f"处理數字 {i} 的圖像 {j} 時出錯: {e}")
    
    return np.array(X), np.array(y), img_list
