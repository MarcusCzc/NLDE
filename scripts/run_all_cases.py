"""
测试脚本：运行文档中提到的四种案例

Case 1: λ₁ = 0, λ₂ = 0  (线性Dirac方程)
Case 2: λ₁ = 0, λ₂ = 1  (仅标量非线性)
Case 3: λ₁ = 1, λ₂ = 0  (仅β非线性)
Case 4: λ₁ = 1, λ₂ = 1  (完整非线性)
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def run_vqa_case(case_num: int, lambda1: float, lambda2: float, output_dir: str):
    """
    运行VQA求解器的特定案例
    
    Args:
        case_num: 案例编号 (1-4)
        lambda1: β非线性系数
        lambda2: 标量非线性系数
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"Running VQA Case {case_num}: λ₁={lambda1}, λ₂={lambda2}")
    print(f"{'='*60}\n")
    
    # 导入必要的模块
    from qiskit.algorithms.optimizers import L_BFGS_B
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    
    # ===== 参数设置 =====
    n = 4       # 减小到4以加快计算
    d = 8       # 减小深度
    time_steps = 20  # 减少步数
    dt = 0.02
    
    # 初始条件参数
    x0 = 0.0
    sigma = 1.0
    k0 = 5.0
    
    # 优化器参数
    initial_runs = 3
    maxiter = 5000
    maxfun = 20000
    ftol = 1e-10
    
    N_constant = 1.0
    
    # ===== 定义辅助函数 =====
    def dirac_initial_condition(x, x0, sigma, k0):
        envelope = np.exp(-((x - x0)**2) / (2*sigma**2)) * np.exp(1j*k0*x)
        norm = np.sqrt(np.sum(np.abs(envelope)**2) * (x[1] - x[0]))
        envelope /= norm
        Psi0 = envelope
        Psi1 = 0.5 * envelope
        return Psi0, Psi1
    
    def ansatz_dirac(qc, n, d, offset, parameters, conj=False):
        n_total = n + 1
        sign = -1 if conj else 1
        for i in range(d + 1):
            for j in range(n_total):
                qc.rx(sign * parameters[2*(i*n_total + j)], j + offset)
                qc.rz(sign * parameters[2*(i*n_total + j) + 1], j + offset)
            if i != d:
                for k in range(n):
                    qc.cx(offset, k + 1 + offset)
                for k in range(n - 1):
                    qc.cx(k + 1 + offset, k + 2 + offset)
                qc.cx(n + offset, 1 + offset)
    
    def stateFromParameters_dirac(ansatz, parameters, n, d):
        n_total = n + 1
        qc = QuantumCircuit(n_total)
        ansatz(qc, n, d, 0, parameters)
        return Statevector(qc)
    
    def apply_linear_step_dirac(psi, n, dt):
        N = 2**n
        psi_reshaped = psi.reshape(2, N)
        
        # β项
        psi_reshaped[0, :] *= np.exp(-1j * dt)
        psi_reshaped[1, :] *= np.exp(1j * dt)
        
        # α项（在动量空间）
        psi_k = np.zeros_like(psi_reshaped, dtype=complex)
        for spin in range(2):
            psi_k[spin, :] = np.fft.fft(psi_reshaped[spin, :])
            psi_k[spin, :] = np.fft.fftshift(psi_k[spin, :])
        
        k_values = np.arange(-N//2, N//2)
        for j in range(N):
            k = k_values[j]
            cos_term = np.cos(k * dt)
            sin_term = np.sin(k * dt)
            psi0_k = psi_k[0, j]
            psi1_k = psi_k[1, j]
            psi_k[0, j] = cos_term * psi0_k - 1j * sin_term * psi1_k
            psi_k[1, j] = -1j * sin_term * psi0_k + cos_term * psi1_k
        
        correction = np.exp(1j * dt * (N//2))
        psi_k *= correction
        
        for spin in range(2):
            psi_k[spin, :] = np.fft.ifftshift(psi_k[spin, :])
            psi_reshaped[spin, :] = np.fft.ifft(psi_k[spin, :])
        
        return psi_reshaped.flatten()
    
    def cost_function_dirac(parameters, parameters_0, ansatz, n, d,
                           lambda1, lambda2, dt, N_constant, dx,
                           startFromStateVector=False):
        N = 2**n
        n_total = n + 1
        
        if startFromStateVector:
            psi_0 = parameters_0
        else:
            psi_0 = np.array(stateFromParameters_dirac(ansatz, parameters_0, n, d))
        
        psi = np.array(stateFromParameters_dirac(ansatz, parameters, n, d))
        psi_tilde = apply_linear_step_dirac(psi_0.copy(), n, dt)
        
        E_ov = np.real(np.vdot(psi, psi_tilde))
        
        a = psi_tilde.reshape(2, N)
        b = psi.reshape(2, N)
        
        E_self = 0.0
        for sigma in range(2):
            for j in range(N):
                term = np.conj(b[sigma, j]) * np.abs(a[sigma, j])**2 * a[sigma, j]
                E_self += np.imag(term)
        
        E_cross = 0.0
        for sigma in range(2):
            for j in range(N):
                term = np.conj(b[sigma, j]) * np.abs(a[1-sigma, j])**2 * a[sigma, j]
                E_cross += np.imag(term)
        
        overlap_term = -2 * E_ov
        nonlinear_term = (dt * N_constant / dx) * (
            (lambda2 + lambda1) * E_self + (lambda2 - lambda1) * E_cross
        )
        
        return overlap_term + nonlinear_term
    
    # ===== 生成初始条件 =====
    N = 2**n
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    dx = x[1] - x[0]
    
    Psi0, Psi1 = dirac_initial_condition(x, x0, sigma, k0)
    initial_condition = np.zeros(2*N, dtype=complex)
    initial_condition[0:N] = Psi0
    initial_condition[N:2*N] = Psi1
    initial_condition /= np.linalg.norm(initial_condition)
    
    # ===== 运行VQA =====
    optimizer = L_BFGS_B(maxiter=maxiter, maxfun=maxfun, ftol=ftol)
    n_total = n + 1
    parameters = np.zeros((time_steps, 2 * n_total * (d + 1)))
    
    # 第一步
    cost_function = lambda x: cost_function_dirac(
        x, initial_condition, ansatz_dirac, n, d,
        lambda1, lambda2, dt, N_constant, dx, True
    )
    
    fvalue_opt = float("Inf")
    for i in range(initial_runs):
        print(f"  Initialization {i+1}/{initial_runs}", end='\r')
        initial_params = 2 * np.pi * np.random.random(2 * n_total * (d + 1))
        sol = optimizer.minimize(cost_function, initial_params)
        fvalue_temp = cost_function(sol.x)
        if fvalue_temp < fvalue_opt:
            fvalue_opt = fvalue_temp
            parameters[0, :] = sol.x
    
    print(f"  Step 1/{time_steps} completed, cost={fvalue_opt:.6e}")
    
    # 后续步骤
    for i in range(1, time_steps):
        cost_function = lambda x: cost_function_dirac(
            x, parameters[i-1, :], ansatz_dirac, n, d,
            lambda1, lambda2, dt, N_constant, dx, False
        )
        sol = optimizer.minimize(cost_function, parameters[i-1, :])
        parameters[i, :] = sol.x
        
        if (i+1) % 5 == 0:
            cost_val = cost_function(sol.x)
            print(f"  Step {i+1}/{time_steps} completed, cost={cost_val:.6e}")
    
    # ===== 提取结果 =====
    psi = np.zeros((time_steps + 1, 2 * N), dtype=complex)
    psi[0, :] = initial_condition
    
    for i in range(time_steps):
        psi[i+1, :] = stateFromParameters_dirac(ansatz_dirac, parameters[i, :], n, d)
    
    psi_up = psi[:, 0:N]
    psi_down = psi[:, N:2*N]
    
    # ===== 可视化 =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 上自旋
    ax = axes[0, 0]
    for i in range(0, time_steps + 1, max(1, time_steps//5)):
        ax.plot(x / np.pi, np.abs(psi_up[i, :])**2,
                label=f"t={i*dt:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_0|^2$")
    ax.set_title(f"Case {case_num}: Spin-up (λ₁={lambda1}, λ₂={lambda2})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 下自旋
    ax = axes[0, 1]
    for i in range(0, time_steps + 1, max(1, time_steps//5)):
        ax.plot(x / np.pi, np.abs(psi_down[i, :])**2,
                label=f"t={i*dt:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_1|^2$")
    ax.set_title("Spin-down")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 总概率密度
    ax = axes[1, 0]
    for i in range(0, time_steps + 1, max(1, time_steps//5)):
        total_prob = np.abs(psi_up[i, :])**2 + np.abs(psi_down[i, :])**2
        ax.plot(x / np.pi, total_prob, label=f"t={i*dt:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_0|^2 + |\\Psi_1|^2$")
    ax.set_title("Total probability density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 自旋极化
    ax = axes[1, 1]
    for i in range(0, time_steps + 1, max(1, time_steps//5)):
        spin_pol = np.abs(psi_up[i, :])**2 - np.abs(psi_down[i, :])**2
        ax.plot(x / np.pi, spin_pol, label=f"t={i*dt:.2f}")
    ax.set_xlabel("$x$ [$\\pi$]")
    ax.set_ylabel("$|\\Psi_0|^2 - |\\Psi_1|^2$")
    ax.set_title("Spin polarization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"case{case_num}_vqa_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Results saved to: {plot_path}")
    
    data_path = os.path.join(output_dir, f"case{case_num}_vqa_data.npz")
    np.savez(data_path,
             psi=psi,
             x=x,
             t=np.linspace(0, time_steps*dt, time_steps+1),
             config={'case': case_num, 'n': n, 'd': d,
                     'lambda1': lambda1, 'lambda2': lambda2,
                     'dt': dt, 'time_steps': time_steps})
    print(f"  Data saved to: {data_path}\n")
    
    plt.close()


def main():
    """运行所有四个案例"""
    output_dir = "/mnt/user-data/outputs"
    
    cases = [
        (1, 0.0, 0.0),  # Case 1: 线性
        (2, 0.0, 1.0),  # Case 2: 仅标量非线性
        (3, 1.0, 0.0),  # Case 3: 仅β非线性
        (4, 1.0, 1.0),  # Case 4: 完整非线性
    ]
    
    print("\n" + "="*60)
    print("DIRAC EQUATION TEST CASES")
    print("="*60)
    print("\nThis script will run all 4 test cases from the documentation:")
    print("  Case 1: λ₁=0, λ₂=0  (Linear Dirac)")
    print("  Case 2: λ₁=0, λ₂=1  (Scalar nonlinearity only)")
    print("  Case 3: λ₁=1, λ₂=0  (β nonlinearity only)")
    print("  Case 4: λ₁=1, λ₂=1  (Full nonlinearity)")
    print("\nNote: Parameters are reduced for faster execution.")
    print("For production runs, increase n, d, time_steps in the script.\n")
    
    for case_num, lambda1, lambda2 in cases:
        try:
            run_vqa_case(case_num, lambda1, lambda2, output_dir)
        except Exception as e:
            print(f"\nError in Case {case_num}: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print("ALL CASES COMPLETED!")
    print("="*60)
    print(f"\nResults saved in: {output_dir}/")
    print("Files generated:")
    for case_num in [1, 2, 3, 4]:
        print(f"  - case{case_num}_vqa_results.png")
        print(f"  - case{case_num}_vqa_data.npz")


if __name__ == "__main__":
    main()
