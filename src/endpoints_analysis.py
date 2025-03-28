import numpy as np
import matplotlib.pyplot as plt

def random_walk_finals(num_steps, num_walks):
    """生成多个二维随机游走的终点位置
    
    通过模拟多次随机游走，每次在x和y方向上随机选择±1移动，
    计算所有随机游走的终点坐标。每个随机游走都从原点(0,0)开始。

    参数:
        num_steps (int): 每次随机游走的步数
        num_walks (int): 随机游走的次数
        
    返回:
        tuple: 包含两个numpy数组的元组 (x_finals, y_finals)
            - x_finals: 所有随机游走终点的x坐标数组，长度为num_walks
            - y_finals: 所有随机游走终点的y坐标数组，长度为num_walks
            
    示例:
        >>> x_finals, y_finals = random_walk_finals(1000, 100)
        >>> print(f"第一个终点坐标: ({x_finals[0]}, {y_finals[0]})")
    """
    x_finals = np.zeros(num_walks)
    y_finals = np.zeros(num_walks)
    for i in range(num_walks):
        x_finals[i] = np.sum(np.random.choice([-1,1],num_steps))
        y_finals[i] = np.sum(np.random.choice([-1,1],num_steps))
    return (x_finals,y_finals)

def plot_endpoints_distribution(endpoints):
    """绘制二维随机游走终点的空间分布散点图
    
    将多次随机游走的终点在二维平面上可视化，观察其空间分布特征。
    图形包含所有终点的散点图，并保持x和y轴的比例相同。

    参数:
        endpoints: 包含x和y坐标的元组 (x_coords, y_coords)
            - x_coords: numpy数组，所有终点的x坐标
            - y_coords: numpy数组，所有终点的y坐标
            
    示例:
        >>> endpoints = random_walk_finals(1000, 1000)
        >>> plot_endpoints_distribution(endpoints)
        >>> plt.show()
    """
    x_coords, y_coords = endpoints  # 直接解包坐标
    plt.scatter(x_coords, y_coords, alpha=0.5)
    plt.axis('equal')
    plt.title('Endpoint Distribution Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')

def analyze_x_distribution(endpoints):
    """分析二维随机游走终点x坐标的统计特性
    
    对随机游走终点的x坐标进行统计分析，计算样本均值和样本方差，
    并通过直方图和理论正态分布曲线可视化其分布特征。理论上，
    大量随机游走的终点x坐标应该服从正态分布。

    参数:
        endpoints: 包含x和y坐标的元组 (x_coords, y_coords)
            - x_coords: numpy数组，所有终点的x坐标
            - y_coords: numpy数组，所有终点的y坐标
    
    返回:
        tuple: (mean, variance)
            - mean (float): x坐标的样本均值
            - variance (float): x坐标的样本方差（使用n-1作为分母）
            
    示例:
        >>> endpoints = random_walk_finals(1000, 1000)
        >>> mean, var = analyze_x_distribution(endpoints)
        >>> print(f"均值: {mean:.2f}, 方差: {var:.2f}")
    """
    x_coords = endpoints[0]  # 获取x坐标数组
    
    # 计算统计量
    mean = np.mean(x_coords) #样本均值
    var = np.var(x_coords, ddof=1) #样本方差
    
    # 绘制x坐标直方图
    plt.hist(x_coords, bins=50, density=True, alpha=0.7)
    
    # 添加理论正态分布曲线
    x = np.linspace(min(x_coords), max(x_coords), 100)
    plt.plot(x, 1/np.sqrt(2*np.pi*var)*np.exp(-(x-mean)**2/(2*var)), 
             'r-', label='Theoretical Normal Distribution')
    
    # 设置图形属性
    plt.title('X-Coordinate Distribution Histogram')
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 打印统计结果
    print(f"Sample mean of X-coordinates: {mean:.2f}")
    print(f"Sample variance of X-coordinates: {var:.2f}")
    

if __name__ == "__main__":
    np.random.seed(42)  # 设置随机种子以保证可重复性
    
    # 生成数据
    endpoints = random_walk_finals(1000, 1000)

    # 创建图形
    plt.figure(figsize=(12, 5))
    
    # 绘制终点分布
    plt.subplot(121)
    plot_endpoints_distribution(endpoints)
    
    # 分析x坐标分布
    plt.subplot(122)
    analyze_x_distribution(endpoints)
    
    plt.tight_layout()
    plt.show()
