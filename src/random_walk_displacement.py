import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class RandomWalkAnalyzer:
    def __init__(self, num_steps=1000, num_walks=1000, seed=None):
        """
        初始化随机行走分析器
        
        参数:
            num_steps (int): 每次行走的步数，默认为1000
            num_walks (int): 行走实验次数，默认为1000  
            seed (int): 随机种子，用于结果复现
        """
        self.num_steps = num_steps
        self.num_walks = num_walks
        self.seed = seed
        self.x_finals = None
        self.y_finals = None
        
    def simulate_walks(self):
        """执行随机行走模拟"""
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # 生成随机步长 (x和y方向)
        steps_x = np.random.choice([-1, 1], size=(self.num_walks, self.num_steps))
        steps_y = np.random.choice([-1, 1], size=(self.num_walks, self.num_steps))
        
        # 计算终点位置
        self.x_finals = np.sum(steps_x, axis=1)
        self.y_finals = np.sum(steps_y, axis=1)
        
    def plot_endpoints(self):
        """绘制终点分布散点图"""
        plt.figure(figsize=(10, 10))
        plt.scatter(self.x_finals, self.y_finals, alpha=0.5, s=10)
        
        # 标记起点和理论中心
        plt.scatter(0, 0, c='red', s=100, label='Origin')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
        
        plt.title(f'Endpoints Distribution ({self.num_walks:,} walks, {self.num_steps:,} steps)')
        plt.xlabel('Final X Position')
        plt.ylabel('Final Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')
        
        # 自动调整坐标轴范围
        max_extent = max(np.abs(self.x_finals).max(), np.abs(self.y_finals).max())
        plt.xlim(-max_extent*1.1, max_extent*1.1)
        plt.ylim(-max_extent*1.1, max_extent*1.1)
        
    def analyze_x_distribution(self):
        """分析x坐标分布"""
        plt.figure(figsize=(12, 6))
        
        # 绘制直方图
        plt.hist(self.x_finals, bins=30, density=True, alpha=0.7, 
                color='blue', edgecolor='black', label='Empirical')
        
        # 绘制理论正态分布曲线
        x = np.linspace(min(self.x_finals), max(self.x_finals), 1000)
        theoretical_std = np.sqrt(self.num_steps)  # 理论标准差
        plt.plot(x, norm.pdf(x, 0, theoretical_std), 'r-', 
                linewidth=2, label=f'Theoretical N(0, {theoretical_std:.1f}²)')
        
        plt.title('X Coordinate Distribution')
        plt.xlabel('Final X Position')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    def print_statistics(self):
        """打印统计信息"""
        print("="*50)
        print("随机行走统计结果")
        print("="*50)
        print(f"实验次数: {self.num_walks:,}")
        print(f"步数: {self.num_steps:,}")
        print("-"*50)
        print(f"x方向均值: {np.mean(self.x_finals):.2f} (理论值: 0.00)")
        print(f"y方向均值: {np.mean(self.y_finals):.2f} (理论值: 0.00)")
        print("-"*50)
        print(f"x方向方差: {np.var(self.x_finals):.2f} (理论值: {self.num_steps:.2f})")
        print(f"y方向方差: {np.var(self.y_finals):.2f} (理论值: {self.num_steps:.2f})")
        print("="*50)

if __name__ == "__main__":
    # 参数设置
    num_steps = 1000    # 每次行走步数
    num_walks = 1000    # 行走实验次数
    seed = 42           # 随机种子
    
    # 创建分析器并执行模拟
    analyzer = RandomWalkAnalyzer(num_steps, num_walks, seed)
    analyzer.simulate_walks()
    
    # 结果分析与可视化
    analyzer.print_statistics()
    
    plt.figure(figsize=(15, 5))
    
    # 绘制终点分布
    plt.subplot(1, 2, 1)
    analyzer.plot_endpoints()
    
    # 分析x坐标分布
    plt.subplot(1, 2, 2)
    analyzer.analyze_x_distribution()
    
    plt.tight_layout()
    plt.show()
