import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.stats import poisson
import time
from matplotlib.ticker import MaxNLocator

def plot_poisson_pmf(lambda_param=8, max_l=None, ax=None):
    """优化后的泊松分布概率质量函数绘图函数
    
    参数:
        lambda_param (float): 泊松分布参数λ
        max_l (int): 最大的l值，自动计算为2λ+10
        ax (matplotlib.axes): 可选的绘图轴对象
        
    返回:
        matplotlib.axes: 绘图轴对象
    """
    if max_l is None:
        max_l = int(2 * lambda_param + 10)
    
    l_values = np.arange(max_l)
    pmf = poisson.pmf(l_values, lambda_param)  # 使用scipy的poisson函数提高精度
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.stem(l_values, pmf, linefmt='b-', markerfmt='bo', basefmt=' ', 
            label=f'Poisson PMF (λ={lambda_param})')
    ax.set_title('Poisson Probability Mass Function', pad=20)
    ax.set_xlabel('Number of successes (l)')
    ax.set_ylabel('Probability p(l)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 确保x轴为整数
    
    return ax

def simulate_coin_flips(n_experiments=10000, n_flips=100, p_head=0.08, seed=None):
    """优化后的抛硬币实验模拟函数
    
    参数:
        n_experiments (int): 实验组数N
        n_flips (int): 每组抛硬币次数
        p_head (float): 正面朝上的概率
        seed (int): 随机种子
        
    返回:
        ndarray: 每组实验中正面朝上的次数
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 向量化操作提高性能
    return np.random.binomial(n_flips, p_head, size=n_experiments)

def compare_simulation_theory(n_experiments=10000, lambda_param=8, seed=None):
    """优化后的实验与理论比较函数
    
    参数:
        n_experiments (int): 实验组数
        lambda_param (float): 泊松分布参数λ
        seed (int): 随机种子
    """
    start_time = time.time()
    
    # 进行实验模拟
    results = simulate_coin_flips(n_experiments, lambda_param/p_head, p_head=0.08, seed=seed)
    
    # 计算理论分布
    max_l = max(int(lambda_param * 2.5), results.max() + 2)
    l_values = np.arange(max_l)
    pmf = poisson.pmf(l_values, lambda_param)
    
    # 创建绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    # 主图: 分布比较
    hist = ax1.hist(results, bins=l_values-0.5, density=True, alpha=0.7,
                   label='Simulation Results', color='skyblue', edgecolor='white')
    ax1.plot(l_values, pmf, 'r-', label='Theoretical Distribution', linewidth=2)
    ax1.set_title(f'Poisson Distribution Comparison (N={n_experiments:,}, λ={lambda_param})', pad=15)
    ax1.set_xlabel('Number of Heads')
    ax1.set_ylabel('Frequency/Probability')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 子图: 相对误差
    hist_values = hist[0]
    bin_centers = l_values[:-1]
    valid_indices = (pmf[:-1] > 0) & (hist_values > 0)
    relative_error = np.zeros_like(pmf[:-1])
    relative_error[valid_indices] = np.abs(hist_values[valid_indices] - pmf[:-1][valid_indices]) / pmf[:-1][valid_indices]
    
    ax2.bar(bin_centers, relative_error, width=0.8, alpha=0.7, color='orange')
    ax2.set_title('Relative Error between Simulation and Theory')
    ax2.set_xlabel('Number of Heads')
    ax2.set_ylabel('Relative Error')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # 计算统计信息
    sim_mean = np.mean(results)
    sim_var = np.var(results)
    elapsed_time = time.time() - start_time
    
    # 打印统计信息
    stats = f"""
    {' Statistics ':=^40}
    Simulation time: {elapsed_time:.2f} seconds
    Experiments: {n_experiments:,}
    {'-'*40}
    Theoretical mean (λ): {lambda_param:.2f}
    Simulation mean: {sim_mean:.2f}
    Absolute error: {abs(sim_mean - lambda_param):.2f}
    {'-'*40}
    Theoretical variance: {lambda_param:.2f}
    Simulation variance: {sim_var:.2f}
    Absolute error: {abs(sim_var - lambda_param):.2f}
    {'='*40}
    """
    print(stats)
    
    return fig

if __name__ == "__main__":
    # 设置全局绘图样式
    plt.style.use('seaborn')
    
    # 1. 绘制理论分布
    plot_poisson_pmf()
    
    # 2&3. 进行实验模拟并比较结果
    compare_simulation_theory(n_experiments=100000, seed=42)
    
    plt.show()
