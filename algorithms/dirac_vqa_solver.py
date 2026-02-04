"""
混合伪谱-变分量子算法求解非线性Dirac方程（Hybrid Pseudo-Spectral VQA-Dirac Solver）

项目说明：
==========
本程序使用混合伪谱-变分量子算法来数值求解非线性Dirac方程

核心思想：
----------
1. 双分量编码：Dirac方程有两个自旋分量，使用 n+1 个量子比特编码（1个自旋比特 + n个位置比特）
2. 参数化表示：使用参数化的量子电路（ansatz）来表示双分量波函数
3. 变分优化：通过优化算法找到最优参数
4. 混合方法：
   - 线性部分：使用伪谱方法（QFT）处理动量算符 α
   - 非线性部分：通过变分优化处理

算法流程：
----------
1. 初始化：生成归一化的初始双分量波函数
2. 时间步进：对每个时间步
   a) 线性子步：在动量空间应用 exp(-i*α*k*dt) * exp(-i*β*dt)
   b) 非线性子步：通过变分优化处理非线性项
3. 状态提取：从优化后的参数中提取量子态
4. 可视化：绘制双分量波函数随时间的演化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, Tuple, Optional
from numpy.typing import NDArray

from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit import QuantumCircuit, transpile, execute, BasicAer
from qiskit.quantum_info import Statevector


# ============================================================================
# 第一部分：解析解函数（初始条件）
# ============================================================================

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
    # 下自旋分量（可以设置不同的初始条件）
    Psi1 = 0.5 * envelope  # 例如，下分量是上分量的一半
    
    return Psi0, Psi1


# ============================================================================
# 第二部分：量子电路构建函数
# ============================================================================

def ansatz_dirac(
    qc: QuantumCircuit,
    n: int,
    d: int,
    offset: int,
    parameters: NDArray[np.floating],
    conj: bool = False
) -> None:
    """
    构建Dirac方程的ansatz量子电路
    
    电路结构：
    - 第一个量子比特：自旋比特（0或1对应Ψ0或Ψ1）
    - 后n个量子比特：位置比特（编码2^n个空间点）
    - 总共 n+1 个量子比特
    
    参数总数：2 * (n+1) * (d+1)
    
    Args:
        qc: 量子电路对象
        n: 位置量子比特数量
        d: 电路深度
        offset: 量子比特起始索引
        parameters: 参数数组，长度为 2*(n+1)*(d+1)
        conj: 是否构建共轭态
    """
    n_total = n + 1  # 总量子比特数（1个自旋 + n个位置）
    sign = -1 if conj else 1

    # 构建 d+1 层旋转门
    for i in range(d + 1):
        # 对每个量子比特应用旋转门
        for j in range(n_total):
            qc.rx(sign * parameters[2*(i*n_total + j)], j + offset)
            qc.rz(sign * parameters[2*(i*n_total + j) + 1], j + offset)

        # 在层之间添加CNOT门（最后一层不需要）
        if i != d:
            # 自旋比特与所有位置比特的CNOT
            for k in range(n):
                qc.cx(offset, k + 1 + offset)  # 自旋比特控制位置比特
            
            # 位置比特之间的链式CNOT
            for k in range(n - 1):
                qc.cx(k + 1 + offset, k + 2 + offset)
            # 最后一个位置比特连回第一个位置比特
            qc.cx(n + offset, 1 + offset)


def stateFromParameters_dirac(
    ansatz: Callable,
    parameters: NDArray[np.floating],
    n: int,
    d: int
) -> Statevector:
    """
    从参数生成Dirac方程的量子态
    
    Args:
        ansatz: ansatz函数
        parameters: 参数数组，长度为 2*(n+1)*(d+1)
        n: 位置量子比特数量
        d: 电路深度
    
    Returns:
        2^(n+1) 维的量子态向量
    """
    n_total = n + 1  # 总量子比特数
    qc = QuantumCircuit(n_total)
    ansatz(qc, n, d, 0, parameters)
    return Statevector(qc)


# ============================================================================
# 第三部分：线性子步的量子电路实现
# ============================================================================

def apply_linear_step_dirac(
    psi: NDArray[np.complexfloating],
    n: int,
    dt: float
) -> NDArray[np.complexfloating]:
    """
    应用Dirac方程的线性子步（使用经典FFT）
    
    线性算符：
    - β 项：在位置空间直接应用 exp(-i*β*dt)
    - α 项：在动量空间应用 exp(-i*α*k*dt)
    
    其中：
    - β = σ_z = diag(1, -1)（对角矩阵）
    - α = σ_x = [[0,1],[1,0]]（非对角矩阵，在动量空间变对角）
    
    Args:
        psi: 2^(n+1) 维量子态向量
        n: 位置量子比特数量
        dt: 时间步长
    
    Returns:
        演化后的量子态向量
    """
    N = 2**n  # 空间点数
    
    # 将一维态向量重塑为 (2, N) 形状
    # psi[0:N] 对应自旋向上 |0⟩⊗|position⟩
    # psi[N:2N] 对应自旋向下 |1⟩⊗|position⟩
    psi_reshaped = psi.reshape(2, N)
    
    # --- Step 1: 应用 β 项（在位置空间）---
    # β = diag(1, -1)，所以 exp(-i*β*dt) = diag(exp(-i*dt), exp(i*dt))
    psi_reshaped[0, :] *= np.exp(-1j * dt)  # 上自旋分量
    psi_reshaped[1, :] *= np.exp(1j * dt)   # 下自旋分量
    
    # --- Step 2: 应用 α 项（在动量空间）---
    # 对每个自旋分量进行FFT
    psi_k = np.zeros_like(psi_reshaped, dtype=complex)
    for spin in range(2):
        psi_k[spin, :] = np.fft.fft(psi_reshaped[spin, :])
        psi_k[spin, :] = np.fft.fftshift(psi_k[spin, :])
    
    # 在动量空间应用 α 算符
    # α 在动量基下的作用是交换两个自旋分量
    # exp(-i*α*k*dt) ≈ cos(k*dt)*I - i*sin(k*dt)*α
    k_values = np.arange(-N//2, N//2)
    
    for j in range(N):
        k = k_values[j]
        # 计算旋转矩阵元素
        cos_term = np.cos(k * dt)
        sin_term = np.sin(k * dt)
        
        # 保存原值
        psi0_k = psi_k[0, j]
        psi1_k = psi_k[1, j]
        
        # 应用旋转：R_x(2*k*dt) 矩阵
        psi_k[0, j] = cos_term * psi0_k - 1j * sin_term * psi1_k
        psi_k[1, j] = -1j * sin_term * psi0_k + cos_term * psi1_k
    
    # 相位校正（类似NLSE中的校正）
    correction = np.exp(1j * dt * (N//2))
    psi_k *= correction
    
    # 逆FFT回位置空间
    for spin in range(2):
        psi_k[spin, :] = np.fft.ifftshift(psi_k[spin, :])
        psi_reshaped[spin, :] = np.fft.ifft(psi_k[spin, :])
    
    # 重塑回一维向量
    return psi_reshaped.flatten()


# ============================================================================
# 第四部分：成本函数（包含三个期望值测量）
# ============================================================================

def cost_function_dirac(
    parameters: NDArray[np.floating],
    parameters_0: Union[NDArray[np.floating], NDArray[np.complexfloating]],
    ansatz: Callable,
    n: int,
    d: int,
    lambda1: float,
    lambda2: float,
    dt: float,
    N_constant: float,
    dx: float,
    startFromStateVector: bool = False
) -> float:
    """
    Dirac方程的成本函数（对应公式30）
    
    C = -2*E_ov + (dt*N/dx) * [(λ1+λ2)*E_self + (λ2-λ1)*E_cross]
    
    其中：
    - E_ov = Re⟨ψ(t+dt)|ψ̃(t)⟩（重叠项）
    - E_self = QNPU_self = Im[∑_{σ,j} b*_{σ,j} |a_{σ,j}|² a_{σ,j}]
    - E_cross = QNPU_cross = Im[∑_{σ,j} b*_{σ,j} |a_{1-σ,j}|² a_{σ,j}]
    
    Args:
        parameters: 当前时刻的参数
        parameters_0: 前一时刻的参数或状态向量
        ansatz: ansatz函数
        n: 位置量子比特数量
        d: 电路深度
        lambda1: 非线性系数1
        lambda2: 非线性系数2
        dt: 时间步长
        N_constant: 归一化常数（通常为1）
        dx: 空间步长
        startFromStateVector: 是否从状态向量开始
    
    Returns:
        成本函数值
    """
    N = 2**n  # 空间点数
    n_total = n + 1  # 总量子比特数
    
    # ===== 步骤1：获取前一时刻的态 =====
    if startFromStateVector:
        psi_0 = parameters_0
    else:
        psi_0 = np.array(stateFromParameters_dirac(ansatz, parameters_0, n, d))
    
    # ===== 步骤2：获取当前试探态 =====
    psi = np.array(stateFromParameters_dirac(ansatz, parameters, n, d))
    
    # ===== 步骤3：应用线性子步 =====
    psi_tilde = apply_linear_step_dirac(psi_0.copy(), n, dt)
    
    # ===== 步骤4：计算三个期望值 =====
    
    # E_ov：重叠项
    E_ov = np.real(np.vdot(psi, psi_tilde))
    
    # 将态重塑为 (2, N) 形状以便计算QNPU
    a = psi_tilde.reshape(2, N)  # a_{σ,j} = ψ̃_σ(x_j, t)
    b = psi.reshape(2, N)        # b_{σ,j} = ψ_σ(x_j, t+dt)
    
    # E_self：自项（每个自旋分量的|a_{σ,j}|²）
    E_self = 0.0
    for sigma in range(2):
        for j in range(N):
            # b*_{σ,j} * |a_{σ,j}|² * a_{σ,j}
            term = np.conj(b[sigma, j]) * np.abs(a[sigma, j])**2 * a[sigma, j]
            E_self += np.imag(term)
    
    # E_cross：交叉项（交换自旋分量的|a_{1-σ,j}|²）
    E_cross = 0.0
    for sigma in range(2):
        for j in range(N):
            # b*_{σ,j} * |a_{1-σ,j}|² * a_{σ,j}
            term = np.conj(b[sigma, j]) * np.abs(a[1-sigma, j])**2 * a[sigma, j]
            E_cross += np.imag(term)
    
    # ===== 步骤5：组合成本函数 =====
    overlap_term = -2 * E_ov
    nonlinear_term = (dt * N_constant / dx) * (
        (lambda2 + lambda1) * E_self + (lambda2 - lambda1) * E_cross
    )
    
    return overlap_term + nonlinear_term


# ============================================================================
# 第五部分：模拟参数设置
# ============================================================================

# ----- 量子电路参数 -----
n = 5       # 位置量子比特数量（2^5 = 32个空间点）
d = 10      # Ansatz电路深度

# ----- 初始条件参数 -----
x0 = 0.0     # 初始位置
sigma = 1.0  # 波包宽度
k0 = 5.0     # 初始动量

# ----- 非线性系数 -----
lambda1 = 1.0  # β 非线性系数
lambda2 = 0.5  # 标量非线性系数

# ----- 时间演化参数 -----
time_steps = 50   # 时间步数
dt = 0.01         # 时间步长

# ----- 优化器参数 -----
initial_runs = 5   # 第一步的重复次数
maxiter = 10000    # 最大迭代次数
maxfun = 50000     # 最大函数评估次数
ftol = 1e-12       # 收敛容差

# ----- 归一化常数 -----
N_constant = 1.0  # 归一化常数（对应文档中的N）

# ===== 生成初始条件 =====
N = 2**n  # 空间点数
x = np.linspace(-np.pi, np.pi, N, endpoint=False)
dx = x[1] - x[0]

Psi0, Psi1 = dirac_initial_condition(x, x0, sigma, k0)

# 组合成量子态向量（2^(n+1)维）
# 前N个元素对应自旋向上，后N个元素对应自旋向下
initial_condition = np.zeros(2*N, dtype=complex)
initial_condition[0:N] = Psi0
initial_condition[N:2*N] = Psi1

# 归一化量子态
initial_condition /= np.linalg.norm(initial_condition)


# ============================================================================
# 第六部分：主模拟循环
# ============================================================================

print("Starting Dirac VQA simulation...")
print(f"Parameters: n={n}, d={d}, N={N}, dt={dt}, steps={time_steps}")
print(f"Nonlinear coefficients: λ1={lambda1}, λ2={lambda2}")
print(f"Initial condition: x0={x0}, σ={sigma}, k0={k0}")

# 初始化优化器
optimizer = L_BFGS_B(maxiter=maxiter, maxfun=maxfun, ftol=ftol)

# 存储参数
n_total = n + 1  # 总量子比特数
parameters = np.zeros((time_steps, 2 * n_total * (d + 1)))

# ===== 第一步：从初始条件开始 =====
cost_function = lambda x: cost_function_dirac(
    x, initial_condition, ansatz_dirac, n, d,
    lambda1, lambda2, dt, N_constant, dx, True
)

fvalue_opt = float("Inf")

for i in range(initial_runs):
    print(f"Step 1/{time_steps} : {i+1}/{initial_runs}")
    
    initial_params = 2 * np.pi * np.random.random(2 * n_total * (d + 1))
    sol = optimizer.minimize(cost_function, initial_params)
    fvalue_temp = cost_function(sol.x)
    
    if fvalue_temp < fvalue_opt:
        fvalue_opt = fvalue_temp
        parameters[0, :] = sol.x
        print(f"  New best cost: {fvalue_opt:.6e}")

# ===== 后续步骤 =====
for i in range(1, time_steps):
    print(f"Step {i+1}/{time_steps}")
    
    cost_function = lambda x: cost_function_dirac(
        x, parameters[i-1, :], ansatz_dirac, n, d,
        lambda1, lambda2, dt, N_constant, dx, False
    )
    
    sol = optimizer.minimize(cost_function, parameters[i-1, :])
    parameters[i, :] = sol.x
    
    if (i+1) % 10 == 0:
        cost_val = cost_function(sol.x)
        print(f"  Cost: {cost_val:.6e}")


# ============================================================================
# 第七部分：提取量子态
# ============================================================================

print("Extracting quantum states...")
psi = np.zeros((time_steps + 1, 2 * N), dtype=complex)
psi[0, :] = initial_condition

for i in range(time_steps):
    psi[i+1, :] = stateFromParameters_dirac(ansatz_dirac, parameters[i, :], n, d)


# ============================================================================
# 第八部分：可视化
# ============================================================================

print("Generating plots...")

# 分离两个自旋分量
psi_up = psi[:, 0:N]      # 自旋向上分量
psi_down = psi[:, N:2*N]  # 自旋向下分量

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 绘制自旋向上分量的概率密度
ax = axes[0, 0]
for i in range(0, time_steps + 1, 10):
    ax.plot(x / np.pi, np.abs(psi_up[i, :])**2, label=f"t={i*dt:.2f}")
ax.set_xlabel("$x$ [$\\pi$]")
ax.set_ylabel("$|\\Psi_0|^2$")
ax.set_title("Spin-up component probability density")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.grid(True, alpha=0.3)

# 绘制自旋向下分量的概率密度
ax = axes[0, 1]
for i in range(0, time_steps + 1, 10):
    ax.plot(x / np.pi, np.abs(psi_down[i, :])**2, label=f"t={i*dt:.2f}")
ax.set_xlabel("$x$ [$\\pi$]")
ax.set_ylabel("$|\\Psi_1|^2$")
ax.set_title("Spin-down component probability density")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.grid(True, alpha=0.3)

# 绘制总概率密度
ax = axes[1, 0]
for i in range(0, time_steps + 1, 10):
    total_prob = np.abs(psi_up[i, :])**2 + np.abs(psi_down[i, :])**2
    ax.plot(x / np.pi, total_prob, label=f"t={i*dt:.2f}")
ax.set_xlabel("$x$ [$\\pi$]")
ax.set_ylabel("$|\\Psi_0|^2 + |\\Psi_1|^2$")
ax.set_title("Total probability density")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.grid(True, alpha=0.3)

# 绘制自旋极化（ρ_j = |Ψ_0|² - |Ψ_1|²）
ax = axes[1, 1]
for i in range(0, time_steps + 1, 10):
    spin_polarization = np.abs(psi_up[i, :])**2 - np.abs(psi_down[i, :])**2
    ax.plot(x / np.pi, spin_polarization, label=f"t={i*dt:.2f}")
ax.set_xlabel("$x$ [$\\pi$]")
ax.set_ylabel("$|\\Psi_0|^2 - |\\Psi_1|^2$")
ax.set_title("Spin polarization")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/dirac_vqa_results.png", dpi=150, bbox_inches='tight')
print("Plot saved to: /home/claude/dirac_vqa_results.png")

# 保存数据
np.savez("/home/claude/dirac_vqa_data.npz",
         parameters=parameters,
         psi=psi,
         x=x,
         t=np.linspace(0, time_steps*dt, time_steps+1),
         config={'n': n, 'd': d, 'lambda1': lambda1, 'lambda2': lambda2,
                 'dt': dt, 'time_steps': time_steps})
print("Data saved to: /home/claude/dirac_vqa_data.npz")

plt.show()

print("Simulation completed!")
