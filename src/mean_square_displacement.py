import numpy as np
import matplotlib.pyplot as plt

def random_walk_finals(num_steps=1000, num_walks=1000):
    """生成多个二维随机游走的终点位置
    
    通过模拟多次随机游走，每次在x和y方向上随机选择±1移动，
    计算所有随机游走的终点坐标。

    参数:
        num_steps (int, optional): 每次随机游走的步数. 默认值为1000
        num_walks (int, optional): 随机游走的次数. 默认值为1000
        
    返回:
        tuple: 包含两个numpy数组的元组 (x_finals, y_finals)
            - x_finals: 所有随机游走终点的x坐标数组
            - y_finals: 所有随机游走终点的y坐标数组
    """
    # 生成随机步长（±1）
    steps_x = np.random.choice([-1, 1], size=(num_walks, num_steps))
    steps_y = np.random.choice([-1, 1], size=(num_walks, num_steps))
    
    # 计算累计位移
    x_finals = np.sum(steps_x, axis=1)
    y_finals = np.sum(steps_y, axis=1)
    
    return x_finals, y_finals

def calculate_mean_square_displacement():
    """计算不同步数下的均方位移
    
    对于预设的步数序列[1000, 2000, 3000, 4000]，分别进行多次随机游走模拟，
    计算每种步数下的均方位移。每次模拟默认进行1000次随机游走取平均。
    
    返回:
        tuple: 包含两个numpy数组的元组 (steps, msd)
            - steps: 步数数组 [1000, 2000, 3000, 4000]
            - msd: 对应的均方位移数组
    """
    step_values = np.array([1000, 2000, 3000, 4000])
    msd_values = np.zeros_like(step_values, dtype=float)
    
    for i, steps in enumerate(step_values):
        x_finals, y_finals = random_walk_finals(steps, 1000)
        # 计算均方位移：<r²> = <x² + y²>
        msd = np.mean(x_finals**2 + y_finals**2)
        msd_values[i] = msd
    
    return step_values, msd_values

def analyze_step_dependence():
    """分析均方位移与步数的关系，并进行最小二乘拟合
    
    返回:
        tuple: (steps, msd, k)
            - steps: 步数数组
            - msd: 对应的均方位移数组
            - k: 拟合得到的比例系数
    """
    steps, msd = calculate_mean_square_displacement()
    
    # 最小二乘拟合 msd = k * steps
    k = np.sum(steps * msd) / np.sum(steps**2)
    
    return steps, msd, k

if __name__ == "__main__":
    np.random.seed(42)  # 设置随机种子保证结果可重复
    
    # 获取数据和拟合结果
    steps, msd, k = analyze_step_dependence()
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制实验数据点
    plt.scatter(steps, msd, color='blue', s=100, label='Simulation Data', zorder=3)
    
    # 绘制理论拟合曲线
    fit_line = k * steps
    plt.plot(steps, fit_line, 'r--', linewidth=2, 
             label=f'Linear Fit: MSD = {k:.4f}·N', zorder=2)
    
    # 设置图形属性
    plt.title('Mean Square Displacement vs Number of Steps', pad=20)
    plt.xlabel('Number of Steps (N)')
    plt.ylabel('Mean Square Displacement (MSD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 打印分析结果
    print("="*50)
    print("随机游走模拟结果分析")
    print("="*50)
    print(f"{'步数(N)':<10}{'模拟MSD':<15}{'理论预测(k*N)':<15}")
    for n, m in zip(steps, msd):
        print(f"{n:<10}{m:<15.2f}{(k*n):<15.2f}")
    print("="*50)
    print(f"拟合得到的比例系数 k = {k:.4f}")
    print(f"理论预期值 k = 2.0 (对于二维随机游走)")
    print("="*50)
    
    plt.tight_layout()
    plt.show()
