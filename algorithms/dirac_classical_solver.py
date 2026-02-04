"""
经典伪谱分裂步法求解非线性Dirac方程
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from numpy.typing import NDArray


def dirac_initial_condition(
    x: NDArray[np.floating],
    x0: float = 0.0,
    sigma: float = 1.0,
    k0: float = 5.0
) -> Tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """
    Dirac方程的初始条件：高斯波包
    
    Args:
        x: 空间位置数组
        x0: 波包中心位置
        sigma: 波包宽度
        k0: 初始动量
    
    Returns:
        (Psi0, Psi1): 两个自旋分量
    """
    # 高斯波包
    envelope = np.exp(-((x - x0)**2) / (2*sigma**2)) * np.exp(1j*k0*x)
    
    # 归一化因子
    norm = np.sqrt(np.sum(np.abs(envelope)**2) * (x[1] - x[0]))
    envelope /= norm
    
    # 上自旋分量
    Psi0 = envelope
    # 下自旋分量
    Psi1 = 0.5 * envelope
    
    return Psi0, Psi1


class DiracClassicalSolver:
    """
    经典伪谱分裂步算法求解非线性Dirac方程
    
    算法步骤：
    1. 线性步（伪谱方法）：
       - 应用 β 项（在位置空间）
       - 应用 α 项（在动量空间）
    2. 非线性 Euler 步
    """

    def __init__(self, n: int = 5, lambda1: float = 1.0, lambda2: float = 0.5, dt: float = 0.01):
        self.n = n
        self.N = 2**n
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
        应用线性演化步骤
        
        对应公式(4)中的线性项：
        - β 项：exp(-i*β*dt)
        - α 项：exp(-i*α*k*dt)（在动量空间）
        
        Args:
            Psi0: 上自旋分量
            Psi1: 下自旋分量
        
        Returns:
            (Psi0_new, Psi1_new): 演化后的两个自旋分量
        """
        # --- Step 1: 应用 β 项（在位置空间）---
        # β = σ_z = diag(1, -1)
        # exp(-i*β*dt) = diag(exp(-i*dt), exp(i*dt))
        Psi0_temp = Psi0 * np.exp(-1j * self.dt)
        Psi1_temp = Psi1 * np.exp(1j * self.dt)
        
        # --- Step 2: 应用 α 项（在动量空间）---
        # α = σ_x，在动量空间变为旋转
        
        # FFT到动量空间
        Psi0_k = np.fft.fftshift(np.fft.fft(Psi0_temp))
        Psi1_k = np.fft.fftshift(np.fft.fft(Psi1_temp))
        
        # 在动量空间应用 α 算符
        # exp(-i*α*k*dt) = cos(k*dt)*I - i*sin(k*dt)*σ_x
        Psi0_k_new = np.zeros_like(Psi0_k)
        Psi1_k_new = np.zeros_like(Psi1_k)
        
        for j in range(self.N):
            k = self.k[j]
            cos_term = np.cos(k * self.dt)
            sin_term = np.sin(k * self.dt)
            
            # 应用旋转矩阵
            Psi0_k_new[j] = cos_term * Psi0_k[j] - 1j * sin_term * Psi1_k[j]
            Psi1_k_new[j] = -1j * sin_term * Psi0_k[j] + cos_term * Psi1_k[j]
        
        # 相位校正
        correction = np.exp(1j * self.dt * (self.N//2))
        Psi0_k_new *= correction
        Psi1_k_new *= correction
        
        # IFFT回位置空间
        Psi0_new = np.fft.ifft(np.fft.ifftshift(Psi0_k_new))
        Psi1_new = np.fft.ifft(np.fft.ifftshift(Psi1_k_new))
        
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
            self.lambda2 * rho * Psi0     # 标量项
        )
        
        # 下自旋分量的非线性演化
        nonlinear_1 = 1j * self.dt * (
            -self.lambda1 * s_z * Psi1 +  # β项对下自旋的贡献（负号）
            self.lambda2 * rho * Psi1      # 标量项
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


def run_simulation(output_dir: str = "/home/claude"):
    """
    运行经典Dirac方程模拟
    """
    # ========== 参数设置 ==========
    n = 5               # 2^5 = 32 points
    x0 = 0.0            # 初始位置
    sigma = 1.0         # 波包宽度
    k0 = 5.0            # 初始动量
    lambda1 = 1.0       # β 非线性系数
    lambda2 = 0.5       # 标量非线性系数
    total_time = 0.5
    num_steps = 50
    dt = total_time / num_steps

    # ========== 初始化求解器和初始条件 ==========
    solver = DiracClassicalSolver(n=n, lambda1=lambda1, lambda2=lambda2, dt=dt)
    x = solver.x
    
    Psi0_init, Psi1_init = dirac_initial_condition(x, x0, sigma, k0)

    print(f"Starting classical Dirac simulation:")
    print(f"  n={n}, N={solver.N}, dt={dt:.6f}, steps={num_steps}")
    print(f"  Initial condition: x0={x0}, σ={sigma}, k0={k0}")
    print(f"  Nonlinear: λ1={lambda1}, λ2={lambda2}")

    # ========== 时间演化 ==========
    Psi0_history, Psi1_history, times = solver.propagate(
        Psi0_init, Psi1_init, num_steps, save_every=1
    )

    print("Simulation completed!")

    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 绘制自旋向上分量
    ax = axes[0, 0]
    for i in range(0, len(times), max(1, len(times)//6)):
        ax.plot(x / np.pi, np.abs(Psi0_history[i, :])**2, label=f"t={times[i]:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_0|^2$")
    ax.set_title("Spin-up component (Classical)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 绘制自旋向下分量
    ax = axes[0, 1]
    for i in range(0, len(times), max(1, len(times)//6)):
        ax.plot(x / np.pi, np.abs(Psi1_history[i, :])**2, label=f"t={times[i]:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_1|^2$")
    ax.set_title("Spin-down component (Classical)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 绘制总概率密度
    ax = axes[1, 0]
    for i in range(0, len(times), max(1, len(times)//6)):
        total_prob = np.abs(Psi0_history[i, :])**2 + np.abs(Psi1_history[i, :])**2
        ax.plot(x / np.pi, total_prob, label=f"t={times[i]:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_0|^2 + |\\Psi_1|^2$")
    ax.set_title("Total probability density (Classical)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 绘制自旋极化
    ax = axes[1, 1]
    for i in range(0, len(times), max(1, len(times)//6)):
        spin_pol = np.abs(Psi0_history[i, :])**2 - np.abs(Psi1_history[i, :])**2
        ax.plot(x / np.pi, spin_pol, label=f"t={times[i]:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_0|^2 - |\\Psi_1|^2$")
    ax.set_title("Spin polarization (Classical)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plot_path = f"{output_dir}/dirac_classical_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # 保存数据
    data_path = f"{output_dir}/dirac_classical_data.npz"
    np.savez(data_path,
             Psi0=Psi0_history,
             Psi1=Psi1_history,
             x=x,
             t=times,
             config={'n': n, 'lambda1': lambda1, 'lambda2': lambda2,
                     'dt': dt, 'num_steps': num_steps})
    print(f"Data saved to: {data_path}")

    plt.show()

    return solver, Psi0_history, Psi1_history


if __name__ == "__main__":
    run_simulation()
