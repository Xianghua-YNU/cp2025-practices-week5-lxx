import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def random_walk_displacement(num_steps, num_simulations):
    """
    模拟随机行走并返回每次模拟的最终位移

    参数:
    num_steps (int): 随机行走的步数
    num_simulations (int): 模拟的次数

    返回:
    numpy.ndarray: 形状为(2, num_simulations)的数组，表示每次模拟的最终位移
    """
    # 检查输入参数的有效性
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("步数必须是正整数")
    if not isinstance(num_simulations, int) or num_simulations <= 0:
        raise ValueError("模拟次数必须是正整数")
    
    # 生成随机步长 (x和y方向各num_simulations次模拟，每次num_steps步)
    steps = np.random.choice([-1, 1], size=(2, num_simulations, num_steps))
    
    # 对步数维度求和得到最终位移
    final_displacements = np.sum(steps, axis=2)
    
    return final_displacements

def plot_displacement_distribution(final_displacements, bins=30):
    """
    绘制位移分布直方图

    参数:
    final_displacements (numpy.ndarray): 形状为(2, num_simulations)的位移数组
    bins (int): 直方图的组数
    """
    # 计算每次模拟的最终位移大小
    displacements = np.sqrt(final_displacements[0]**2 + final_displacements[1]**2)
    
    plt.figure(figsize=(10, 6))
    plt.hist(displacements, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # 添加理论Rayleigh分布曲线
    sigma = np.sqrt(num_steps)  # 理论标准差
    x = np.linspace(0, displacements.max(), 1000)
    rayleigh = (x / sigma**2) * np.exp(-x**2 / (2 * sigma**2))
    plt.plot(x, rayleigh, 'r-', linewidth=2, label='Rayleigh Distribution')
    
    plt.title(f'Final Displacement Distribution (N={num_steps:,} steps)')
    plt.xlabel('Displacement Magnitude')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_displacement_square_distribution(final_displacements, bins=30):
    """
    绘制位移平方分布直方图

    参数:
    final_displacements (numpy.ndarray): 形状为(2, num_simulations)的位移数组
    bins (int): 直方图的组数
    """
    # 计算位移平方
    displacement_squares = final_displacements[0]**2 + final_displacements[1]**2
    
    plt.figure(figsize=(10, 6))
    plt.hist(displacement_squares, bins=bins, density=True, alpha=0.7, color='green', edgecolor='black')
    
    # 添加理论指数分布曲线
    mean_square = np.mean(displacement_squares)
    x = np.linspace(0, displacement_squares.max(), 1000)
    exponential = (1 / (2 * num_steps)) * np.exp(-x / (2 * num_steps))
    plt.plot(x, exponential, 'r-', linewidth=2, label='Exponential Distribution')
    
    plt.title(f'Displacement Square Distribution (N={num_steps:,} steps)')
    plt.xlabel('Squared Displacement')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # 参数设置
    num_steps = 1000    # 随机行走的步数
    num_simulations = 1000  # 模拟的次数
    bins = 30           # 直方图的组数
    
    np.random.seed(42)  # 设置随机种子保证结果可重复
    
    # 获取模拟结果
    final_displacements = random_walk_displacement(num_steps, num_simulations)
    
    # 打印统计信息
    print("="*50)
    print("随机游走模拟统计结果")
    print("="*50)
    print(f"模拟次数: {num_simulations:,}")
    print(f"步数: {num_steps:,}")
    print(f"x方向平均位移: {np.mean(final_displacements[0]):.2f} (理论值: 0.00)")
    print(f"y方向平均位移: {np.mean(final_displacements[1]):.2f} (理论值: 0.00)")
    print(f"x方向位移方差: {np.var(final_displacements[0]):.2f} (理论值: {num_steps:.2f})")
    print(f"y方向位移方差: {np.var(final_displacements[1]):.2f} (理论值: {num_steps:.2f})")
    print("="*50)
    
    # 绘制位移分布直方图
    plot_displacement_distribution(final_displacements, bins)
    
    # 绘制位移平方分布直方图
    plot_displacement_square_distribution(final_displacements, bins)
