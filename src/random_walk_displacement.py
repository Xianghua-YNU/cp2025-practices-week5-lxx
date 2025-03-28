import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, expon

def random_walk_displacement(num_steps, num_simulations, seed=None):
    """
    模拟随机行走并返回每次模拟的最终位移

    参数:
        num_steps (int): 随机行走的步数
        num_simulations (int): 模拟的次数
        seed (int, optional): 随机种子

    返回:
        numpy.ndarray: 形状为(2, num_simulations)的数组，表示每次模拟的最终位移
    """
    # 检查输入参数的有效性
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("步数必须是正整数")
    if not isinstance(num_simulations, int) or num_simulations <= 0:
        raise ValueError("模拟次数必须是正整数")
    
    if seed is not None:
        np.random.seed(seed)
    
    # 生成随机步长 (x和y方向各num_simulations次模拟，每次num_steps步)
    steps = np.random.choice([-1, 1], size=(2, num_simulations, num_steps))
    
    # 对步数维度求和得到最终位移
    return np.sum(steps, axis=2)

def plot_displacement_distribution(final_displacements, num_steps, bins=30):
    """
    绘制位移分布直方图

    参数:
        final_displacements (numpy.ndarray): 形状为(2, N)的位移数组
        num_steps (int): 随机行走步数（用于标题显示）
        bins (int): 直方图的组数
    """
    # 计算每次模拟的最终位移大小
    displacements = np.linalg.norm(final_displacements, axis=0)
    
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(displacements, bins=bins, density=True, 
                              alpha=0.7, color='blue', edgecolor='black')
    
    # 添加理论Rayleigh分布曲线
    scale = np.sqrt(num_steps)  # 理论尺度参数
    x = np.linspace(0, displacements.max(), 1000)
    plt.plot(x, rayleigh.pdf(x, scale=scale), 'r-', 
             linewidth=2, label=f'Rayleigh(σ={scale:.1f})')
    
    plt.title(f'Final Displacement Distribution (N={num_steps:,} steps)')
    plt.xlabel('Displacement Magnitude')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_displacement_square_distribution(final_displacements, num_steps, bins=30):
    """
    绘制位移平方分布直方图

    参数:
        final_displacements (numpy.ndarray): 形状为(2, N)的位移数组
        num_steps (int): 随机行走步数（用于标题显示）
        bins (int): 直方图的组数
    """
    # 计算位移平方
    displacement_squares = np.sum(final_displacements**2, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.hist(displacement_squares, bins=bins, density=True, 
            alpha=0.7, color='green', edgecolor='black')
    
    # 添加理论指数分布曲线
    scale = 2 * num_steps  # 理论尺度参数
    x = np.linspace(0, displacement_squares.max(), 1000)
    plt.plot(x, expon.pdf(x, scale=scale), 'r-', 
             linewidth=2, label=f'Exponential(λ={1/scale:.4f})')
    
    plt.title(f'Displacement Square Distribution (N={num_steps:,} steps)')
    plt.xlabel('Squared Displacement')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

def print_statistics(final_displacements, num_steps):
    """打印统计信息"""
    print("="*50)
    print("随机游走模拟统计结果")
    print("="*50)
    print(f"模拟次数: {final_displacements.shape[1]:,}")
    print(f"步数: {num_steps:,}")
    
    for i, axis in enumerate(['x', 'y']):
        disp = final_displacements[i]
        print(f"{axis}方向平均位移: {np.mean(disp):.2f} (理论值: 0.00)")
        print(f"{axis}方向位移方差: {np.var(disp):.2f} (理论值: {num_steps:.2f})")
        print("-"*50)
    
    # 计算位移大小统计量
    displacements = np.linalg.norm(final_displacements, axis=0)
    print(f"平均位移大小: {np.mean(displacements):.2f}")
    print(f"位移大小方差: {np.var(displacements):.2f}")
    print("="*50)

if __name__ == "__main__":
    # 参数设置
    num_steps = 1000      # 随机行走的步数
    num_simulations = 10000  # 模拟的次数
    bins = 50             # 直方图的组数
    seed = 42             # 随机种子
    
    # 获取模拟结果
    final_displacements = random_walk_displacement(
        num_steps, num_simulations, seed=seed)
    
    # 打印统计信息
    print_statistics(final_displacements, num_steps)
    
    # 创建图形
    plt.figure(figsize=(14, 6))
    
    # 绘制位移分布直方图
    plt.subplot(1, 2, 1)
    plot_displacement_distribution(final_displacements, num_steps, bins)
    
    # 绘制位移平方分布直方图
    plt.subplot(1, 2, 2)
    plot_displacement_square_distribution(final_displacements, num_steps, bins)
    
    plt.tight_layout()
    plt.show()
