import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom, expon

def generate_coin_sequence(n_flips, p_head=0.08):
    """生成硬币序列，1表示正面，0表示反面"""
    return np.random.choice([0, 1], size=n_flips, p=[1-p_head, p_head])

def calculate_waiting_times(coin_sequence):
    """计算两次正面之间的等待时间（反面次数）"""
    head_indices = np.nonzero(coin_sequence)[0]  # 获取所有正面位置
    if len(head_indices) < 2:
        return np.array([])  # 不足两个正面无法计算等待时间
    intervals = np.diff(head_indices)  # 计算连续正面之间的间隔
    return intervals - 1  # 减去1得到中间的反面数量

def plot_waiting_time_histogram(waiting_times, log_scale=False, n_flips=None):
    """绘制等待时间直方图"""
    plt.figure(figsize=(10, 6))
    
    # 自动确定合适的bins数量
    max_wait = np.max(waiting_times) if len(waiting_times) > 0 else 10
    bins = min(50, max_wait)
    
    # 绘制直方图
    plt.hist(waiting_times, bins=bins, density=True, alpha=0.7, 
             color='blue', edgecolor='black')
    
    # 设置标题和标签
    title = "Waiting Time Distribution"
    if n_flips is not None:
        title += f" (n={n_flips:,})"
    plt.title(title, fontsize=14)
    plt.xlabel("Waiting Time (Number of Tails Between Heads)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    
    if log_scale:
        plt.yscale('log')
        plt.title(title + " (Log Scale)", fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.show()
def analyze_waiting_time(waiting_times, p_head=0.08):
    """分析等待时间的统计特性"""
    stats = {}
    
    if len(waiting_times) > 0:
        stats["mean"] = np.mean(waiting_times)
        stats["std"] = np.std(waiting_times)
    else:
        stats["mean"] = stats["std"] = np.nan
    
    # 理论值计算
    stats["theoretical_mean"] = (1 - p_head) / p_head  # 几何分布均值
    stats["exponential_mean"] = 1 / p_head  # 指数分布均值
    
    return stats

def run_experiment(n_flips, title, p_head=0.08):
    """运行一次等待时间实验"""
    print("="*60)
    print(f"{title:^60}")
    print("="*60)
    
    # 生成序列并计算等待时间
    sequence = generate_coin_sequence(n_flips, p_head)
    waiting_times = calculate_waiting_times(sequence)
    
    # 分析统计特性
    stats = analyze_waiting_time(waiting_times, p_head)
    
    # 打印结果
    print(f"Number of heads: {len(np.nonzero(sequence)[0]):,}")
    print(f"Number of waiting times: {len(waiting_times):,}")
    print("-"*60)
    print(f"Experimental mean waiting time: {stats['mean']:.2f}")
    print(f"Theoretical mean (Geometric): {stats['theoretical_mean']:.2f}")
    print(f"Theoretical mean (Exponential): {stats['exponential_mean']:.2f}")
    print("-"*60)
    print(f"Experimental standard deviation: {stats['std']:.2f}")
    print("="*60)
    
    # 绘制直方图
    plt.figure(figsize=(15, 5))
    
    # 普通坐标直方图
    plt.subplot(1, 2, 1)
    plot_waiting_time_histogram(waiting_times, n_flips=n_flips)
    
    # 对数坐标直方图
    plt.subplot(1, 2, 2)
    plot_waiting_time_histogram(waiting_times, log_scale=True, n_flips=n_flips)
    
    plt.tight_layout()
    plt.show()
    
    return waiting_times, stats

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 任务一：1000次抛掷
    waiting_times_1k, stats_1k = run_experiment(1000, "Task 1: 1000 Coin Flips")
    
    # 任务二：1000000次抛掷
    print("\n")
    waiting_times_1m, stats_1m = run_experiment(1000000, "Task 2: 1,000,000 Coin Flips")
