import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def random_walk_finals(num_steps, num_walks):
    """
    模拟随机行走并返回每次模拟的最终位移
    
    参数:
        num_steps (int): 随机行走的步数
        num_walks (int): 模拟的次数
        
    返回:
        tuple: (x_finals, y_finals) 两个一维数组，表示x和y方向的最终位移
    """
    # 处理边界情况
    if num_steps == 0:
        return np.zeros(num_walks), np.zeros(num_walks)
    if num_walks == 0:
        return np.array([]), np.array([])
    
    # 生成随机步长 (x和y方向各num_walks次模拟，每次num_steps步)
    steps_x = np.random.choice([-1, 1], size=num_walks * num_steps).reshape(num_walks, num_steps)
    steps_y = np.random.choice([-1, 1], size=num_walks * num_steps).reshape(num_walks, num_steps)
    
    # 对步数维度求和得到最终位移
    x_finals = np.sum(steps_x, axis=1)
    y_finals = np.sum(steps_y, axis=1)
    
    return x_finals, y_finals

def plot_endpoints_distribution(endpoints):
    """绘制二维随机游走终点的空间分布散点图"""
    x_coords, y_coords = endpoints
    
    plt.scatter(x_coords, y_coords, alpha=0.5, s=10, color='blue')
    plt.title(f'Endpoints Distribution (N={len(x_coords):,})')
    plt.xlabel('Final X Position')
    plt.ylabel('Final Y Position')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')

def analyze_x_distribution(endpoints):
    """分析二维随机游走终点x坐标的统计特性"""
    x_coords, _ = endpoints
    
    # 计算统计量
    mean = np.mean(x_coords)
    variance = np.var(x_coords, ddof=1)
    
    # 绘制直方图
    plt.hist(x_coords, bins=30, density=True, alpha=0.7, 
             color='green', label='Empirical Distribution')
    
    # 绘制理论正态分布曲线
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 100)
    theoretical_std = np.sqrt(len(x_coords))  # 修正理论标准差计算
    p = norm.pdf(x, 0, theoretical_std)
    
    plt.plot(x, p, 'r-', linewidth=2, 
             label=f'Theoretical N(0, {theoretical_std:.1f}²)')
    
    plt.title('X Coordinate Distribution')
    plt.xlabel('Final X Position')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(f"统计结果:")
    print(f"样本均值: {mean:.2f} (理论值: 0.00)")
    print(f"样本方差: {variance:.2f} (理论值: {len(x_coords):.2f})")
    print(f"样本标准差: {np.sqrt(variance):.2f} (理论值: {np.sqrt(len(x_coords)):.2f})")
    
    return mean, variance

if __name__ == "__main__":
    np.random.seed(42)
    
    num_steps = 1000
    num_walks = 1000
    
    endpoints = random_walk_finals(num_steps, num_walks)

    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plot_endpoints_distribution(endpoints)
    
    plt.subplot(1, 2, 2)
    analyze_x_distribution(endpoints)
    
    plt.tight_layout()
    plt.show()
