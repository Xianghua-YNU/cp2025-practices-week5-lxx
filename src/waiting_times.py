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
    plt.show()  # 添加这行以显示图形
