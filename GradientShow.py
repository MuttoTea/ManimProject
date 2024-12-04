import torch  
import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.animation import FuncAnimation  

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义三个目标函数  

def himmelblau(x1, x2):  
    """  
    Himmelblau 函数  
    """  
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 -7)**2  

def rastrigin(x1, x2):  
    """  
    Rastrigin 函数  
    """  
    return 20 + x1**2 + x2**2 - 10 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2))  

def ackley(x1, x2):  
    """  
    Ackley 函数  
    """  
    return -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x1**2 + x2**2))) - torch.exp(0.5 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2))) + 20 + torch.e  

def rosenbrock(x1, x2, a=1, b=100):  
    return (a - x1)**2 + b * (x2 - x1**2)**2  

def booth(x1, x2):  
    return (x1 + 2 * x2 - 7)**2 + (2 * x1 + x2 - 5)**2  

def beale(x1, x2):  
    term1 = (1.5 - x1 + x1 * x2)**2  
    term2 = (2.25 - x1 + x1 * x2**2)**2  
    term3 = (2.625 - x1 + x1 * x2**3)**2  
    return term1 + term2 + term3  

def schaffer(x1, x2):  
    return 0.5 + ((x1**2 + x2**2)**2 - 25 * (x1**2 + x2**2) + 100) / (1 + 0.001 * (x1**2 + x2**2)**2)  

def matyas(x1, x2):  
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2  

def easom(x1, x2):  
    return -torch.cos(x1) * torch.cos(x2) * torch.exp(-( (x1 - torch.pi)**2 + (x2 - torch.pi)**2 ))  

def styblinski_tang(x1, x2):  
    return 0.5 * (x1**4 - 16 * x1**2 + 5 * x1 + x2**4 - 16 * x2**2 + 5 * x2)

# 梯度下降算法实现  

def gradient_descent_pytorch(func, start_point, learning_rate, max_iters, tolerance):  
    """  
    使用 PyTorch 进行梯度下降算法寻找函数最小值  
    :param func: 目标函数，接受两个 PyTorch 张量 x1 和 x2  
    :param start_point: 初始点 [x1, x2]  
    :param learning_rate: 学习率  
    :param max_iters: 最大迭代次数  
    :param tolerance: 梯度的容忍度（停止条件）  
    :return: 记录了每次迭代的 [x1, x2, f(x1, x2)] 的列表  
    """  
    # 初始化 x1 和 x2 为可训练的参数  
    x = torch.tensor(start_point, dtype=torch.float32, requires_grad=True)  

    path = []  # 存储迭代路径  
    # 计算初始函数值  
    y = func(x[0], x[1]).item()  
    path.append([x[0].item(), x[1].item(), y])  
    print(f"初始位置: x1 = {x[0].item():.6f}, x2 = {x[1].item():.6f}, f(x1,x2) = {y:.6f}")  

    for i in range(1, max_iters + 1):  
        # 清零梯度  
        if x.grad is not None:  
            x.grad.zero_()  
        
        # 计算函数值  
        y = func(x[0], x[1])  
        
        # 反向传播计算梯度  
        y.backward()  

        # 获取梯度  
        grad = x.grad.detach().clone()  
        
        # 更新参数  
        with torch.no_grad():  
            x -= learning_rate * grad  

        # 记录新的位置和函数值  
        new_x1, new_x2 = x[0].item(), x[1].item()  
        new_y = func(x[0], x[1]).item()  
        path.append([new_x1, new_x2, new_y])  
        print(f"迭代 {i}: x1 = {new_x1:.6f}, x2 = {new_x2:.6f}, f(x1,x2) = {new_y:.6f}")  

        # 检查梯度的范数是否小于容忍度  
        grad_norm = torch.norm(grad).item()  
        if grad_norm < tolerance:  
            print(f"梯度足够小 ({grad_norm:.6f} < {tolerance}), 停止迭代。")  
            break  

    return np.array(path)  

# 三维图像绘制函数  

def plot_3d_gradient_descent(path, func, func_name):  
    """  
    绘制目标函数的三维曲面及梯度下降路径  
    :param path: 记录了每次迭代的 [x1, x2, y] 的数组  
    :param func: 目标函数，用于绘制曲面  
    :param func_name: 目标函数名称，用于标题显示  
    """  
    fig = plt.figure(figsize=(10, 7))  
    ax = fig.add_subplot(111, projection='3d')  

    # 设置绘图范围  
    x1_min, x1_max = np.min(path[:,0]) - 2, np.max(path[:,0]) + 2  
    x2_min, x2_max = np.min(path[:,1]) - 2, np.max(path[:,1]) + 2  

    # 创建网格以绘制曲面  
    X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))  
    
    # 将 X1 和 X2 转换为 Torch 张量以计算 Z  
    X1_tensor = torch.tensor(X1, dtype=torch.float32)  
    X2_tensor = torch.tensor(X2, dtype=torch.float32)  
    with torch.no_grad():  
        Z_tensor = func(X1_tensor, X2_tensor)  
    Z = Z_tensor.numpy()  

    # 绘制曲面  
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)  

    # 绘制梯度下降路径  
    ax.plot(path[:,0], path[:,1], path[:,2], color='r', marker='o', label='梯度下降路径')  

    # 标注初始点和终点  
    ax.scatter(path[0,0], path[0,1], path[0,2], color='blue', s=100, label='初始点')  
    ax.scatter(path[-1,0], path[-1,1], path[-1,2], color='green', s=100, label='最小点')  

    # 设置标签和标题  
    ax.set_xlabel('x1')  
    ax.set_ylabel('x2')  
    ax.set_zlabel('f(x1, x2)')  
    ax.set_title(f'梯度下降三维可视化（{func_name}）')  
    ax.legend()  

    plt.show()  

# 动画绘制函数（可选）  

def animate_gradient_descent_3d(path, func, func_name, interval=100):  
    """  
    动态显示梯度下降过程中的点移动及路径（3D动画）  
    :param path: 记录了每次迭代的 [x1, x2, y] 的数组  
    :param func: 目标函数，用于绘制曲面  
    :param func_name: 目标函数名称，用于标题显示  
    :param interval: 动画每帧之间的时间间隔（毫秒）  
    """  
    fig = plt.figure(figsize=(10, 7))  
    ax = fig.add_subplot(111, projection='3d')  

    # 设置绘图范围  
    x1_min, x1_max = np.min(path[:,0]) - 2, np.max(path[:,0]) + 2  
    x2_min, x2_max = np.min(path[:,1]) - 2, np.max(path[:,1]) + 2  

    # 创建网格以绘制曲面  
    X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))  
    X1_tensor = torch.tensor(X1, dtype=torch.float32)  
    X2_tensor = torch.tensor(X2, dtype=torch.float32)  
    with torch.no_grad():  
        Z_tensor = func(X1_tensor, X2_tensor)  
    Z = Z_tensor.numpy()  

    # 绘制曲面  
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)  

    # 初始化动画元素  
    path_line, = ax.plot([], [], [], color='r', marker='o', label='梯度下降路径')  
    current_point, = ax.plot([], [], [], 'ro', markersize=8)  
    txt = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)  

    # 标注初始点  
    ax.scatter(path[0,0], path[0,1], path[0,2], color='blue', s=100, label='初始点')  

    # 设置标签和标题  
    ax.set_xlabel('x1')  
    ax.set_ylabel('x2')  
    ax.set_zlabel('f(x1, x2)')  
    ax.set_title(f'梯度下降三维动画（{func_name}）')  
    ax.legend()  

    def init():  
        path_line.set_data([], [])  
        path_line.set_3d_properties([])  
        current_point.set_data([], [])  
        current_point.set_3d_properties([])  
        txt.set_text('')  
        return path_line, current_point, txt  

    def update(frame):  
        x1, x2, y = path[frame]  
        path_line.set_data(path[:frame+1, 0], path[:frame+1, 1])  
        path_line.set_3d_properties(path[:frame+1, 2])  
        current_point.set_data([x1], [x2])  
        current_point.set_3d_properties([y])  
        txt.set_text(f"迭代: {frame}\nx1 = {x1:.4f}\nx2 = {x2:.4f}\nf(x1,x2) = {y:.4f}")  
        return path_line, current_point, txt  

    ani = FuncAnimation(  
        fig,  
        update,  
        frames=len(path),  
        init_func=init,  
        blit=False,  
        interval=interval,  
        repeat=False  
    )  

    plt.show()  

# 主程序  

def main():  
    # 定义函数字典  
    functions = {  
    '1': {'func': himmelblau, 'name': 'Himmelblau 函数'},  
    '2': {'func': rastrigin, 'name': 'Rastrigin 函数'},  
    '3': {'func': ackley, 'name': 'Ackley 函数'},  
    '4': {'func': rosenbrock, 'name': 'Rosenbrock 函数'},  
    '5': {'func': booth, 'name': 'Booth 函数'},  
    '6': {'func': beale, 'name': 'Beale 函数'},  
    '7': {'func': schaffer, 'name': 'Schaffer 函数'},  
    '8': {'func': matyas, 'name': 'Matyas 函数'},  
    '9': {'func': easom, 'name': 'Easom 函数'},  
    '10': {'func': styblinski_tang, 'name': 'Styblinski-Tang 函数'}  
    }

    # 用户选择目标函数  
    print("请选择用于演示的目标函数：")  
    print("1. Himmelblau 函数")  
    print("2. Rastrigin 函数")  
    print("3. Ackley 函数")  
    print("4. Rosenbrock 函数")  
    print("5. Booth 函数")  
    print("6. Beale 函数")  
    print("7. Schaffer 函数")  
    print("8. Matyas 函数")  
    print("9. Easom 函数")  
    print("10. Styblinski-Tang 函数")
    choice = input("请输入数字（1~10）：").strip()  

    if choice not in functions:  
        print("无效的选择。程序将退出。")  
        return  

    selected_func = functions[choice]['func']  
    func_name = functions[choice]['name']  

    # 设置梯度下降参数  
    start_point = [0.0, 0.0]    # 初始点 [x1, x2]  
    learning_rate = 0.01        # 学习率，根据需要调整  
    max_iters = 1000            # 最大迭代次数  
    tolerance = 1e-6            # 梯度容忍度  

    print(f"\n您选择了 {func_name}。\n")  

    # 执行梯度下降，获取路径  
    path = gradient_descent_pytorch(selected_func, start_point, learning_rate, max_iters, tolerance)  

    # 绘制三维静态图像  
    plot_3d_gradient_descent(path, selected_func, func_name)  

    # 动态显示梯度下降过程的三维动画（可选）  
    animate_choice = input("是否需要查看梯度下降的三维动画？（y/n）：").strip().lower()  
    if animate_choice == 'y':  
        animate_gradient_descent_3d(path, selected_func, func_name, interval=100)  
    else:  
        print("动画已跳过。")  

if __name__ == "__main__":  
    main()