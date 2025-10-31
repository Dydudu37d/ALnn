import numpy as np
from ALnn import MultiLogicNet, MiniCNN, load_num_data

print("测试ALnn库...")
print(f"库版本: {__import__('ALnn').__version__}")

# 测试MultiLogicNet的基本功能
def test_multi_logic_net():
    print("\n测试多逻辑门网络...")
    # 创建简单的测试数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # AND, OR, XOR, NAND 真值表
    y = np.array([[0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0]])
    
    # 初始化网络
    model = MultiLogicNet(hidden_size=4, activation='tanh')
    print(f"网络结构: {model.input_size} -> {model.hidden_size} -> {model.output_size}")
    
    # 进行一次前向传播测试
    try:
        a1, a2 = model.forward(X)
        print("前向传播测试成功!")
        print(f"预测输出形状: {a2.shape}")
        return True
    except Exception as e:
        print(f"前向传播测试失败: {e}")
        return False

# 测试MiniCNN的基本功能
def test_mini_cnn():
    print("\n测试迷你卷积神经网络...")
    # 创建一个简单的28x28测试图像
    test_image = np.random.rand(28, 28)
    
    # 初始化CNN
    cnn = MiniCNN()
    print(f"卷积核数量: {len(cnn.conv_filters)}")
    
    # 进行一次前向传播测试
    try:
        probabilities, pool_results = cnn.forward(test_image)
        print("前向传播测试成功!")
        print(f"输出概率形状: {probabilities.shape}")
        print(f"池化结果数量: {len(pool_results)}")
        return True
    except Exception as e:
        print(f"前向传播测试失败: {e}")
        return False

if __name__ == "__main__":
    test_multi_logic_net()
    test_mini_cnn()
    print("\n测试完成!")