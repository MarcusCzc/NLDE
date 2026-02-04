"""
归一化经典伪谱分裂步法求解非线性Dirac方程
（每个时间步后手动归一化）
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
    """
    envelope = np.exp(-((x - x0)**2) / (2*sigma**2)) * np.exp(1j*k0*x)
    norm = np.sqrt(np.sum(np.abs(envelope)**2) * (x[1] - x[0]))
    envelope /= norm
    
    Psi0 = envelope
    Psi1 = 0.5 * envelope
    
    return Psi0, Psi1


def normalize_dirac_wavefunction(
    Psi0: NDArray,
    Psi1: NDArray,
    N_constant: float,
    dx: float
) -> Tuple[NDArray, NDArray]:
    """
    归一化Dirac双分量波函数
    
    使其满足：∫(|Ψ0|² + |Ψ1|²) dx = N_constant
    
    Args:
        Psi0: 上自旋分量
        Psi1: 下自旋分量
        N_constant: 归一化常数（通常为1）
        dx: 空间步长
    
    Returns:
        (Psi0_norm, Psi1_norm): 归一化后的两个分量
    """
    # 计算当前归一化
    current_norm_squared = np.sum(np.abs(Psi0)**2 + np.abs(Psi1)**2) * dx
    
    # 目标归一化
    target_norm_squared = N_constant
    
    # 计算归一化因子
    if current_norm_squared > 0:
        scale = np.sqrt(target_norm_squared / current_norm_squared)
        return Psi0 * scale, Psi1 * scale
    else:
        return Psi0, Psi1


class DiracNormalizedClassicalSolver:
    """
    归一化经典伪谱分裂步算法求解非线性Dirac方程
    
    与普通经典算法的唯一区别：每个时间步后手动归一化
    """

    def __init__(
        self,
        n: int = 5,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        dt: float = 0.01,
        N_constant: float = 1.0
    ):
        self.n = n
        self.N = 2**n
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dt = dt
        self.N_constant = N_constant
        
        # 空间网格
        self.x = np.linspace(-np.pi, np.pi, self.N, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        
        # 波数
        self.k = np.arange(-self.N//2, self.N//2)

    def linear_step(self, Psi0: NDArray, Psi1: NDArray) -> Tuple[NDArray, NDArray]:
        """应用线性演化步骤"""
        # β 项
        Psi0_temp = Psi0 * np.exp(-1j * self.dt)
        Psi1_temp = Psi1 * np.exp(1j * self.dt)
        
        # α 项（在动量空间）
        Psi0_k = np.fft.fftshift(np.fft.fft(Psi0_temp))
        Psi1_k = np.fft.fftshift(np.fft.fft(Psi1_temp))
        
        Psi0_k_new = np.zeros_like(Psi0_k)
        Psi1_k_new = np.zeros_like(Psi1_k)
        
        for j in range(self.N):
            k = self.k[j]
            cos_term = np.cos(k * self.dt)
            sin_term = np.sin(k * self.dt)
            
            Psi0_k_new[j] = cos_term * Psi0_k[j] - 1j * sin_term * Psi1_k[j]
            Psi1_k_new[j] = -1j * sin_term * Psi0_k[j] + cos_term * Psi1_k[j]
        
        # 相位校正
        correction = np.exp(1j * self.dt * (self.N//2))
        Psi0_k_new *= correction
        Psi1_k_new *= correction
        
        # IFFT
        Psi0_new = np.fft.ifft(np.fft.ifftshift(Psi0_k_new))
        Psi1_new = np.fft.ifft(np.fft.ifftshift(Psi1_k_new))
        
        return Psi0_new, Psi1_new

    def nonlinear_step(self, Psi0: NDArray, Psi1: NDArray) -> Tuple[NDArray, NDArray]:
        """应用非线性演化步骤"""
        rho = np.abs(Psi0)**2 + np.abs(Psi1)**2
        s_z = np.abs(Psi0)**2 - np.abs(Psi1)**2
        
        nonlinear_0 = 1j * self.dt * (
            self.lambda1 * s_z * Psi0 +
            self.lambda2 * rho * Psi0
        )
        
        nonlinear_1 = 1j * self.dt * (
            -self.lambda1 * s_z * Psi1 +
            self.lambda2 * rho * Psi1
        )
        
        Psi0_new = Psi0 + nonlinear_0
        Psi1_new = Psi1 + nonlinear_1
        
        return Psi0_new, Psi1_new

    def step(self, Psi0: NDArray, Psi1: NDArray) -> Tuple[NDArray, NDArray]:
        """
        执行一个完整的时间步（线性步 + 非线性步 + 归一化）
        """
        # 线性步
        Psi0_tilde, Psi1_tilde = self.linear_step(Psi0, Psi1)
        
        # 非线性步
        Psi0_new, Psi1_new = self.nonlinear_step(Psi0_tilde, Psi1_tilde)
        
        # **关键区别：手动归一化**
        Psi0_new, Psi1_new = normalize_dirac_wavefunction(
            Psi0_new, Psi1_new, self.N_constant, self.dx
        )
        
        return Psi0_new, Psi1_new

    def propagate(
        self,
        Psi0_init: NDArray,
        Psi1_init: NDArray,
        num_steps: int,
        save_every: int = 1
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """传播波函数多个时间步"""
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
    运行归一化经典Dirac方程模拟
    """
    # ========== 参数设置 ==========
    n = 5
    x0 = 0.0
    sigma = 1.0
    k0 = 5.0
    lambda1 = 1.0
    lambda2 = 0.5
    total_time = 0.5
    num_steps = 50
    dt = total_time / num_steps
    N_constant = 1.0

    # ========== 初始化求解器和初始条件 ==========
    solver = DiracNormalizedClassicalSolver(
        n=n, lambda1=lambda1, lambda2=lambda2, dt=dt, N_constant=N_constant
    )
    x = solver.x
    
    Psi0_init, Psi1_init = dirac_initial_condition(x, x0, sigma, k0)

    print(f"Starting normalized classical Dirac simulation:")
    print(f"  n={n}, N={solver.N}, dt={dt:.6f}, steps={num_steps}")
    print(f"  Initial condition: x0={x0}, σ={sigma}, k0={k0}")
    print(f"  Nonlinear: λ1={lambda1}, λ2={lambda2}")
    print(f"  Normalization constant: {N_constant}")

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
    ax.set_title("Spin-up component (Normalized Classical)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 绘制自旋向下分量
    ax = axes[0, 1]
    for i in range(0, len(times), max(1, len(times)//6)):
        ax.plot(x / np.pi, np.abs(Psi1_history[i, :])**2, label=f"t={times[i]:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_1|^2$")
    ax.set_title("Spin-down component (Normalized Classical)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 绘制总概率密度
    ax = axes[1, 0]
    for i in range(0, len(times), max(1, len(times)//6)):
        total_prob = np.abs(Psi0_history[i, :])**2 + np.abs(Psi1_history[i, :])**2
        ax.plot(x / np.pi, total_prob, label=f"t={times[i]:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_0|^2 + |\\Psi_1|^2$")
    ax.set_title("Total probability density (Normalized Classical)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 绘制自旋极化
    ax = axes[1, 1]
    for i in range(0, len(times), max(1, len(times)//6)):
        spin_pol = np.abs(Psi0_history[i, :])**2 - np.abs(Psi1_history[i, :])**2
        ax.plot(x / np.pi, spin_pol, label=f"t={times[i]:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_0|^2 - |\\Psi_1|^2$")
    ax.set_title("Spin polarization (Normalized Classical)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plot_path = f"{output_dir}/dirac_normalized_classical_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # 保存数据
    data_path = f"{output_dir}/dirac_normalized_classical_data.npz"
    np.savez(data_path,
             Psi0=Psi0_history,
             Psi1=Psi1_history,
             x=x,
             t=times,
             config={'n': n, 'lambda1': lambda1, 'lambda2': lambda2,
                     'dt': dt, 'num_steps': num_steps, 'N_constant': N_constant})
    print(f"Data saved to: {data_path}")

    plt.show()

    return solver, Psi0_history, Psi1_history


if __name__ == "__main__":
    run_simulation()
