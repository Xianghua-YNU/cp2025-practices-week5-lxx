import matplotlib.pyplot as plt
import numpy as np

def random_walk_2d(steps, seed=None):
    """生成二维随机行走轨迹
    
    参数:
        steps (int): 随机行走的步数
        seed (int): 随机种子（可选）
        
    返回:
        tuple: 包含x和y坐标序列的元组 (x_coords, y_coords)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成随机步长（±1）
    x_steps = np.random.choice([-1, 1], size=steps)
    y_steps = np.random.choice([-1, 1], size=steps)
    
    # 计算累积和得到轨迹
    x_coords = np.cumsum(x_steps)
    y_coords = np.cumsum(y_steps)
    
    return x_coords, y_coords

def plot_single_walk(path, title="2D Random Walk"):
    """绘制单个随机行走轨迹
    
    参数:
        path (tuple): 包含x和y坐标序列的元组
        title (str): 图标题
    """
    x_coords, y_coords = path
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, 'b-', alpha=0.7, linewidth=1, label='Path')
    plt.scatter(x_coords[0], y_coords[0], c='green', s=100, label='Start', zorder=3)
    plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, label='End', zorder=3)
    
    plt.title(title, fontsize=14)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 自动调整坐标轴范围
    margin = max(max(abs(x_coords)), max(abs(y_coords))) * 0.1
    plt.xlim(min(x_coords)-margin, max(x_coords)+margin)
    plt.ylim(min(y_coords)-margin, max(y_coords)+margin)

def plot_multiple_walks(steps=1000):
    """在2x2子图中绘制四个不同的随机行走轨迹"""
    plt.figure(figsize=(12, 12))
    
    for i in range(4):
        ax = plt.subplot(2, 2, i+1)
        path = random_walk_2d(steps, seed=i+42)  # 不同种子产生不同轨迹
        
        x, y = path
        ax.plot(x, y, 'b-', alpha=0.7, linewidth=1)
        ax.scatter(x[0], y[0], c='green', s=80, label='Start')
        ax.scatter(x[-1], y[-1], c='red', s=80, label='End')
        
        ax.set_title(f'Random Walk {i+1} ({steps} steps)', fontsize=11)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 自动调整坐标轴范围
        margin = max(max(abs(x)), max(abs(y))) * 0.1
        ax.set_xlim(min(x)-margin, max(x)+margin)
        ax.set_ylim(min(y)-margin, max(y)+margin)
    
    plt.tight_layout(pad=3.0)

if __name__ == "__main__":
    # 参数设置
    num_steps = 1000  # 随机行走步数
    
    # 1. 生成并绘制单个轨迹
    single_path = random_walk_2d(num_steps, seed=42)
    plot_single_walk(single_path, f"2D Random Walk ({num_steps} steps)")
    
    # 2. 生成并绘制多个轨迹
    plot_multiple_walks(num_steps)
    
    plt.show()
