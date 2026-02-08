"""
经典伪谱分裂步法求解非线性 Dirac 方程
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from numpy.typing import NDArray


def dirac_initial_condition(
    x: NDArray[np.floating],
    x0: float = 0.0,
    sigma: float = 1.0,
    k0: float = 5.0,
    m: float = 1.0
) -> Tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """
    Dirac 正能量高斯波包初始条件（适合伪谱法）
    """

    dx = x[1] - x[0]

    # 空间包络
    g = np.exp(-((x - x0)**2) / (2*sigma**2))

    # 动量相位
    phase = np.exp(1j * k0 * x)

    # Dirac 正能量自旋子
    E0 = np.sqrt(k0**2 + m**2)
    u0 = np.sqrt((E0 + m) / (2 * E0))
    u1 = np.sign(k0) * np.sqrt((E0 - m) / (2 * E0))

    Psi0 = u0 * g * phase
    Psi1 = u1 * g * phase

    # L2 归一化
    norm = np.sqrt(np.sum(np.abs(Psi0)**2 + np.abs(Psi1)**2) * dx)
    Psi0 /= norm
    Psi1 /= norm

    return Psi0.astype(np.complex128), Psi1.astype(np.complex128)


class DiracClassicalSolver:
    """
    经典伪谱分裂步算法求解非线性 Dirac 方程
    
    算法步骤：
    1. 线性步（伪谱方法）：
       - FFT 到动量空间 → 应用 α 项（exp(-iαkΔt)）
       - IFFT 回位置空间 → 应用 β 项（exp(-iβmΔt)）
    2. 非线性 Euler 步
    """

    def __init__(self, n: int = 6, m: float = 1.0, lambda1: float = 1.0, lambda2: float = 0.5, dt: float = 0.01):
        self.n = n
        self.N = 2**n
        self.m = m
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dt = dt

        # 空间网格：x ∈ [-π, π)
        self.x = np.linspace(-np.pi, np.pi, self.N, endpoint=False)
        self.dx = self.x[1] - self.x[0]

        # 波数：k_j = -N/2 + j, j = 0, ..., N-1
        self.k = np.arange(-self.N//2, self.N//2)

    def linear_step(self, Psi0: NDArray, Psi1: NDArray) -> Tuple[NDArray, NDArray]:
        """
        线性演化步骤 - 严格实现论文公式(4):
        Ψ_new = exp(-iβmΔt) * F^{-1}[ exp(-iαkΔt) * F[Ψ] ]
    
        该格式分裂为两个可精确计算的子步：
        1. FFT 到动量空间 → 应用 exp(-iαkΔt) (α = σ_x，在动量空间是对角化的)
        2. IFFT 回位置空间 → 应用 exp(-iβmΔt) (β = σ_z，在位置空间是对角化的)
    
        Args:
            Psi0: 上自旋分量波函数 (位置空间)
            Psi1: 下自旋分量波函数 (位置空间)
    
        Returns:
            Psi0_new: 演化后的上自旋分量
            Psi1_new: 演化后的下自旋分量
        """

        # 获取系统参数：格点数 N 和时间步长 Δt
        N, dt = self.N, self.dt

        # 波数数组 k：预计算好的整数波数 [-N/2, -N/2+1, ..., N/2-1]
        k = self.k  
        
        # 获取质量
        m = self.m

        # --- 步骤 1: 傅里叶变换到动量空间 ---
        # 1.1 执行 FFT 将波函数变换到动量空间
        #     标准 FFT 返回的波数顺序是 [0, 1, ..., N/2-1, -N/2, ..., -1]
        Psi0_k = np.fft.fft(Psi0)
        Psi1_k = np.fft.fft(Psi1)

        # 1.2 使用 fftshift 将波数重新排序为对称顺序
        #     顺序变为 [-N/2, -N/2+1, ..., -1, 0, 1, ..., N/2-1]
        Psi0_k = np.fft.fftshift(Psi0_k)
        Psi1_k = np.fft.fftshift(Psi1_k)

        # --- 步骤 2: 动量空间演化 exp(-iαkΔt) ---
        # 2.1 计算旋转矩阵的三角函数系数
        #     对于每个波数 k，我们需要计算 exp(-iσ_x·kΔt)
        #     矩阵指数展开：exp(-iσ_x·kΔt) = cos(kΔt)·I - i·sin(kΔt)·σ_x
        coskt = np.cos(k * dt)  # 单位矩阵系数
        sinkt = np.sin(k * dt)  # σ_x 矩阵系数

        # 2.2 应用 2×2 旋转矩阵到每个动量模式
        #     [ Ψ0_k_new ]   [  cos(kΔt)   -i·sin(kΔt) ] [ Ψ0_k ]
        #     [ Ψ1_k_new ] = [ -i·sin(kΔt)   cos(kΔt)   ] [ Ψ1_k ]
        #     这是对每个 k 的独立 2×2 矩阵乘法
        Psi0_k_new = coskt * Psi0_k - 1j * sinkt * Psi1_k
        Psi1_k_new = -1j * sinkt * Psi0_k + coskt * Psi1_k

        # --- 步骤 3: 逆傅里叶变换回位置空间 ---
        # 3.1 使用 ifftshift 将波数顺序恢复为标准 FFT 顺序
        #     从对称顺序变回 [0, 1, ..., N/2-1, -N/2, ..., -1]
        Psi0_k_new = np.fft.ifftshift(Psi0_k_new)
        Psi1_k_new = np.fft.ifftshift(Psi1_k_new)

        # 3.2 执行逆 FFT，返回位置空间波函数
        Psi0_tmp = np.fft.ifft(Psi0_k_new)
        Psi1_tmp = np.fft.ifft(Psi1_k_new)

        # --- 步骤 4: 位置空间演化 exp(-iβΔt) ---
        # 4.1 β = σ_z = diag(1, -1)，所以 exp(-iβΔt) 是对角矩阵
        #     exp(-iσ_z·Δt) = diag( exp(-iΔt), exp(iΔt) )
        #     因为 σ_z 的特征值是 +1 和 -1
        # 4.2 分别对两个分量应用相位旋转
        #     上分量（对应 σ_z 特征值 +1）：乘以 exp(-iΔt)
        #     下分量（对应 σ_z 特征值 -1）：乘以 exp(+iΔt)
        Psi0_new = Psi0_tmp * np.exp(-1j * m * dt)
        Psi1_new = Psi1_tmp * np.exp(+1j * m * dt)

        # 返回演化后的波函数
        return Psi0_new, Psi1_new

    def nonlinear_step(self, Psi0: NDArray, Psi1: NDArray) -> Tuple[NDArray, NDArray]:
        """
        应用非线性演化步骤（Euler方法）
        
        对应公式(6)中的非线性项
        
        Args:
            Psi0: 上自旋分量（线性步之后）
            Psi1: 下自旋分量（线性步之后）
        
        Returns:
            (Psi0_new, Psi1_new): 演化后的两个自旋分量
        """

        # 计算局域密度
        rho = np.abs(Psi0)**2 + np.abs(Psi1)**2  # 总密度
        s_z = np.abs(Psi0)**2 - np.abs(Psi1)**2  # 自旋极化

        # 非线性项（对应公式5和6）
        # F_j = I + i*dt*[(λ1*(ψ†βψ)β + λ2*‖ψ‖²*I)]

        # 上自旋分量的非线性演化
        nonlinear_0 = 1j * self.dt * (
            self.lambda1 * s_z * Psi0 +  # β项对上自旋的贡献
            self.lambda2 * rho * Psi0    # 标量项
        )

        # 下自旋分量的非线性演化
        nonlinear_1 = 1j * self.dt * (
            -self.lambda1 * s_z * Psi1 +  # β项对下自旋的贡献（负号）
            self.lambda2 * rho * Psi1     # 标量项
        )

        Psi0_new = Psi0 + nonlinear_0
        Psi1_new = Psi1 + nonlinear_1

        return Psi0_new, Psi1_new

    def step(self, Psi0: NDArray, Psi1: NDArray) -> Tuple[NDArray, NDArray]:
        """
        执行一个完整的时间步（线性步 + 非线性步）
        
        Args:
            Psi0: 当前上自旋分量
            Psi1: 当前下自旋分量
        
        Returns:
            (Psi0_new, Psi1_new): 演化后的两个自旋分量
        """

        # 线性步
        Psi0_tilde, Psi1_tilde = self.linear_step(Psi0, Psi1)

        # 非线性步
        Psi0_new, Psi1_new = self.nonlinear_step(Psi0_tilde, Psi1_tilde)

        return Psi0_new, Psi1_new

    def propagate(
        self,
        Psi0_init: NDArray,
        Psi1_init: NDArray,
        num_steps: int,
        save_every: int = 1
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        传播波函数多个时间步
        
        Args:
            Psi0_init: 初始上自旋分量
            Psi1_init: 初始下自旋分量
            num_steps: 时间步数
            save_every: 每隔多少步保存一次
        
        Returns:
            (Psi0_history, Psi1_history, times): 演化历史和时间数组
        """

        Psi0 = Psi0_init.copy()
        Psi1 = Psi1_init.copy()

        save_indices = sorted(set(list(range(0, num_steps, save_every)) + [num_steps]))
        Psi0_history = []
        Psi1_history = []
        times = []
        current_save_idx = 0

        for step in range(num_steps + 1):
            t = step * self.dt
            if current_save_idx < len(save_indices) and step == save_indices[current_save_idx]:
                Psi0_history.append(Psi0.copy())
                Psi1_history.append(Psi1.copy())
                times.append(t)
                current_save_idx += 1
            if step < num_steps:
                Psi0, Psi1 = self.step(Psi0, Psi1)

        return np.array(Psi0_history), np.array(Psi1_history), np.array(times)


def run_single_case(lambda1: float, lambda2: float, n: int = 6, num_steps: int = 50):
    """
    运行单个 lambda 组合的模拟
    
    Args:
        lambda1: β 非线性系数
        lambda2: 标量非线性系数
        n: 网格大小参数 (2^n points)
        num_steps: 时间步数
    
    Returns:
        (x, Psi0_history, Psi1_history, times): 空间网格、波函数历史和时间数组
    """

    # ========== 参数设置 ==========
    x0 = 0.0            # 初始位置
    sigma = 0.3           # 波包宽度
    k0 = 5.0            # 初始动量
    m = 1.0             # 质量参数
    total_time = 0.5
    dt = total_time / num_steps

    # ========== 初始化求解器和初始条件 ==========
    solver = DiracClassicalSolver(n=n, m=m, lambda1=lambda1, lambda2=lambda2, dt=dt)
    x = solver.x
    
    Psi0_init, Psi1_init = dirac_initial_condition(x, x0, sigma, k0, m)

    # ========== 时间演化 ==========
    Psi0_history, Psi1_history, times = solver.propagate(
        Psi0_init, Psi1_init, num_steps, save_every=1
    )

    return x, Psi0_history, Psi1_history, times


def run_simulation(output_dir: str = "results"):
    """
    运行四个不同 lambda 组合的经典 Dirac 方程模拟，并绘制总概率密度分布
    """

    # ========== 参数设置 ==========
    n = 8               # 2^8 = 256 points
    num_steps = 50      # 总时间步数
    time_steps_to_plot = [1, 10, 20, 40, 50]  # 要绘制的时间步

    # 四个lambda组合
    lambda_cases = [
        (0.0, 0.0, "λ₁=0, λ₂=0"),
        (0.0, 1.0, "λ₁=0, λ₂=1"),
        (1.0, 0.0, "λ₁=1, λ₂=0"),
        (1.0, 1.0, "λ₁=1, λ₂=1")
    ]

    print("Starting classical Dirac simulations for 4 lambda cases...")
    print(f"  n={n}, N={2**n}, steps={num_steps}")
    print(f"  Time steps to plot: {time_steps_to_plot}")

    # ========== 运行所有四个案例 ==========
    results = []
    for lambda1, lambda2, label in lambda_cases:
        print(f"\nRunning case: {label}")
        x, Psi0_history, Psi1_history, times = run_single_case(
            lambda1=lambda1, lambda2=lambda2, n=n, num_steps=num_steps
        )
        results.append((x, Psi0_history, Psi1_history, times, label))
        print(f"  Case {label} completed!")

    print("\nAll simulations completed!")

    # ========== 可视化：四个子图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (x, Psi0_history, Psi1_history, times, label) in enumerate(results):
        ax = axes[idx]

        # 从完整模拟结果中提取并绘制指定时间步的总概率密度
        for step in time_steps_to_plot:
            if step <= num_steps and step < len(times):
                total_prob = np.abs(Psi0_history[step, :])**2 + np.abs(Psi1_history[step, :])**2
                ax.plot(x / np.pi, total_prob, label=f"t={step}", linewidth=1.5)

        ax.set_xlabel("$x$ [$\\pi$]", fontsize=11)
        ax.set_ylabel("$|\\Psi_0|^2 + |\\Psi_1|^2$", fontsize=11)
        ax.set_title(f"Total probability density: {label}", fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = f"{output_dir}/dirac_classical_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    plt.show()

    return results


if __name__ == "__main__":
    run_simulation()
