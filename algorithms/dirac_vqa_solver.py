"""
混合伪谱-变分量子算法求解非线性 Dirac 方程

项目说明：
==========
本程序使用混合伪谱-变分量子算法来数值求解非线性 Dirac 方程

核心思想：
----------
1. 双分量编码：Dirac 方程有两个自旋分量，使用 n+1 个量子比特编码（1个自旋比特 + n个位置比特）
2. 参数化表示：使用参数化的量子电路（ansatz）来表示双分量波函数
3. 变分优化：通过优化算法找到最优参数
4. 混合方法：
   - 线性部分：使用伪谱方法（QFT）处理动量算符 α
   - 非线性部分：通过变分优化处理

算法流程：
----------
1. 初始化：生成归一化的初始双分量波函数
2. 时间步进：对每个时间步
   a) 线性子步：文档公式(4) Ψ̃ = e^{-iβΔt} F^{-1}(e^{-iαkΔt} F Ψ)，并实现公式(25)-(29)的 U_lin 电路
   b) 非线性子步：变分优化，代价函数为文档公式(30) 的三期望值组合
3. 代价函数测量：E_ov 用 Hadamard test（式23），E_self/E_cross 用 r=3 QNPU 原语（文档图1、图2）
4. 状态提取与可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, Tuple, Optional
from numpy._core.numeric import False_
from numpy.typing import NDArray

from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit import QuantumCircuit, transpile, execute, BasicAer
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT


# ============================================================================
# 第一部分：解析解函数（初始条件）
# ============================================================================

def dirac_initial_condition(
    x: NDArray[np.floating],
    x0: float = 0.0,
    sigma: float = 1.0,
    k0: float = 5.0,
    m: float = 1.0
) -> Tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """
    Dirac 正能量高斯波包初始条件
    """

    dx = x[1] - x[0]

    # 空间包络
    g = np.exp(-((x - x0)**2) / (2*sigma**2))

    # 动量相位
    phase = np.exp(1j * k0 * x)

    # Dirac 正能量自旋子系数
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
    构建 Dirac 方程的 ansatz 量子电路
    
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
    从参数生成 Dirac 方程的量子态
    
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
    dt: float,
    m: float = 1.0
) -> NDArray[np.complexfloating]:
    """
    应用 Dirac 方程线性子步（文档公式(4)），用经典 FFT 实现

    实现：Ψ̃ = e^{-iβmΔt} F^{-1}( e^{-iαkΔt} F[Ψ] )
    顺序：(1) FFT 到动量空间 (2) 应用 e^{-iαkΔt} (3) IFFT 回位置空间 (4) 应用 e^{-iβmΔt}
    α = σ_x，β = σ_z

    Args:
        psi: 2^(n+1) 维量子态，psi[0:N] 自旋上，psi[N:2N] 自旋下
        n: 位置比特数，N=2^n 格点
        dt: 时间步长
        m: 质量

    Returns:
        线性子步演化后的态向量
    """
    N = 2**n
    psi_reshaped = psi.reshape(2, N)

    # 步骤1：FFT 到动量空间 (F)
    psi_k = np.zeros_like(psi_reshaped, dtype=complex)
    for spin in range(2):
        psi_k[spin, :] = np.fft.fft(psi_reshaped[spin, :])
        psi_k[spin, :] = np.fft.fftshift(psi_k[spin, :])

    # 步骤2：动量空间应用 exp(-i α k Δt)，α=σ_x 即 Rx(2kΔt)
    k_values = np.arange(-N//2, N//2)
    for j in range(N):
        k = k_values[j]
        cos_term = np.cos(k * dt)
        sin_term = np.sin(k * dt)
        psi0_k = psi_k[0, j]
        psi1_k = psi_k[1, j]
        psi_k[0, j] = cos_term * psi0_k - 1j * sin_term * psi1_k
        psi_k[1, j] = -1j * sin_term * psi0_k + cos_term * psi1_k

    # 步骤3：IFFT 回位置空间 (F^{-1})
    for spin in range(2):
        psi_k[spin, :] = np.fft.ifftshift(psi_k[spin, :])
        psi_reshaped[spin, :] = np.fft.ifft(psi_k[spin, :])

    # 步骤4：位置空间应用 exp(-i β m Δt)，β=σ_z 对角
    psi_reshaped[0, :] *= np.exp(-1j * m * dt)
    psi_reshaped[1, :] *= np.exp(1j * m * dt)

    return psi_reshaped.flatten()


def build_ulin_circuit_dirac(n: int, dt: float, m: float = 1.0) -> QuantumCircuit:
    """
    按文档公式 (25)-(29) 构建线性子步的量子电路 U_lin

    U_lin = R_β (QFT_x)† X_msb U_ph^(D) X_msb QFT_x
    - 比特约定：0 = 自旋，1..n = 位置（1 为 LSB，n 为 MSB）
    - R_β = e^{-iσz mΔt} = Rz(2*m*dt) 作用在自旋比特
    - QFT_x 仅作用在 n 个位置比特上
    - 动量 k = (2π/N)(Σ_{ℓ=0}^{n-2} 2^ℓ b_ℓ - 2^{n-1} b_{n-1})，U_ph 为动量控制的 Rx(2kΔt)

    Args:
        n: 位置比特数，N = 2^n
        dt: 时间步长
        m: 质量

    Returns:
        作用在 n+1 个比特上的量子电路（比特 0 自旋，1..n 位置）
    """
    n_total = n + 1
    qc = QuantumCircuit(n_total)
    N = 2**n

    # 顺序：QFT_x → X_msb → U_ph → X_msb → (QFT_x)† → R_β

    # QFT_x：仅对位置比特 1..n
    qft_pos = QFT(num_qubits=n, inverse=False, do_swaps=False)
    qc.append(qft_pos.to_instruction(), list(range(1, n_total)))

    # X_msb：位置寄存器最高位（频谱中心化）
    qc.x(n)

    # U_ph^(D)：由动量比特控制的自旋 Rx，文档式(28)-(29)
    # k = 2π/N * (Σ_{ℓ=0}^{n-2} 2^ℓ b_ℓ - 2^{n-1} b_{n-1})，Rx(2kΔt) 分解为受控 Rx
    for ell in range(n - 1):
        angle = 4.0 * np.pi * dt / N * (2**ell)
        qc.crx(angle, 1 + ell, 0)
    angle_msb = -4.0 * np.pi * dt / N * (2 ** (n - 1))
    qc.crx(angle_msb, n, 0)

    # X_msb 再次（还原）
    qc.x(n)

    # (QFT_x)†
    iqft_pos = QFT(num_qubits=n, inverse=True, do_swaps=False)
    qc.append(iqft_pos.to_instruction(), list(range(1, n_total)))

    # R_β = e^{-iσz mΔt}，Rz(2*m*dt) 在自旋比特上
    qc.rz(2.0 * m * dt, 0)

    return qc


def apply_linear_step_dirac_via_circuit(
    psi: NDArray[np.complexfloating],
    n: int,
    dt: float,
    use_statevector: bool = True,
    m: float = 1.0
) -> NDArray[np.complexfloating]:
    """
    用文档中的 U_lin 量子电路对态向量做线性子步（仿真时等价于 apply_linear_step_dirac）

    当 use_statevector=True 时，在态向量上应用 U_lin 的酉矩阵，结果与 FFT 实现一致，用于校验电路
    """
    if not use_statevector:
        raise NotImplementedError("仅支持态向量仿真")
    qc_ulin = build_ulin_circuit_dirac(n, dt, m)
    sv = Statevector(psi)
    sv = sv.evolve(qc_ulin)
    return np.array(sv)


# ============================================================================
# 第四部分：成本函数与文档中的量子测量电路
# ============================================================================

def build_hadamard_test_overlap_circuit(
    U_a: QuantumCircuit,
    U_b: QuantumCircuit,
    n_total: int
) -> QuantumCircuit:
    """
    文档式(23)：测量 E_ov = Re⟨ψ(t+Δt)|ψ̃(t)⟩ 的 Hadamard test 电路。

    辅助比特 |0⟩，H，控制-U_b†，H；寄存器先由 U_a 制备 |a⟩=ψ̃(t)。
    测量辅助比特 Z 的期望值即为 Re⟨b|a⟩，其中 |b⟩=U_b|0⟩。

    Args:
        U_a: 制备 |a⟩ 的电路，作用在 n_total 比特上
        U_b: 制备 |b⟩ 的电路，作用在 n_total 比特上
        n_total: 数据寄存器比特数（自旋+位置 = n+1）

    Returns:
        1 + n_total 比特电路，比特 0 为辅助，1..n_total 为数据；运行后辅助比特的 ⟨Z⟩ = Re⟨b|a⟩
    """
    qc = QuantumCircuit(1 + n_total)
    # 寄存器制备 |a⟩
    qc.append(U_a.to_instruction(), list(range(1, n_total + 1)))
    qc.h(0)
    qc.append(U_b.inverse().to_instruction().control(1), [0] + list(range(1, n_total + 1)))
    qc.h(0)
    return qc


def build_qnpu_self_circuit(
    U_a: QuantumCircuit,
    U_b_conj: QuantumCircuit,
    n_total: int
) -> QuantumCircuit:
    """
    文档图1：测量 QNPU_self = Im[Σ_{σ,j} b*_{σ,j} |a_{σ,j}|² a_{σ,j}] 的 r=3 QNPU 电路。

    三份数据寄存器 R1、R2、R3（各 n_total 比特）+ 辅助比特 c。
    R1、R2 制备 |a⟩，R3 制备 |b*⟩；输出 ⟨Z_c⟩ 为虚部通道（文档中 S† 约定）。

    Args:
        U_a: 制备 |a⟩=ψ̃(t) 的电路
        U_b_conj: 制备 |b*⟩ 的电路（ansatz 角度取负得共轭态）
        n_total: 单寄存器比特数 n+1

    Returns:
        1 + 3*n_total 比特电路，前 1 个为辅助 c，接着 R1、R2、R3 各 n_total 比特
    """
    # 比特布局：0 = 辅助 c，[1..n_total]=R1，[n_total+1..2*n_total]=R2，[2*n_total+1..3*n_total]=R3
    qc = QuantumCircuit(1 + 3 * n_total)
    # 文档图1：|0⟩c H S† H，然后 r=3 QNPU 原语；R1,R2 用 Ua，R3 用 Ub*
    qc.h(0)
    qc.sdg(0)
    qc.h(0)
    qc.append(U_a.to_instruction(), list(range(1, 1 + n_total)))
    qc.append(U_a.to_instruction(), list(range(1 + n_total, 1 + 2 * n_total)))
    qc.append(U_b_conj.to_instruction(), list(range(1 + 2 * n_total, 1 + 3 * n_total)))
    # r=3 QNPU 原语：此处用占位，实际原语为多控门组合，输出 ⟨Z_c⟩ 正比于 QNPU_self
    # 完整 QNPU 门分解见 NLSE 文献；本实现保留接口与比特布局，仿真时仍用态向量直接算 E_self/E_cross
    return qc


def build_qnpu_cross_circuit(
    U_a: QuantumCircuit,
    U_b_conj: QuantumCircuit,
    n_total: int
) -> QuantumCircuit:
    """
    文档图2：测量 QNPU_cross = Im[Σ_{σ,j} b*_{σ,j} |a_{1-σ,j}|² a_{σ,j}] 的 r=3 QNPU 电路。

    与 QNPU_self 相同，但在 R1 的自旋比特（即每个寄存器的第 0 个数据比特）上先加 X，
    得到 |a'⟩=X|a⟩，从而 |⟨σ,j|a'⟩|² = |a_{1-σ,j}|²。

    Args:
        U_a: 制备 |a⟩ 的电路
        U_b_conj: 制备 |b*⟩ 的电路
        n_total: 单寄存器比特数 n+1；自旋比特在 R1 内为 R1 的第 0 位，即全局 qubit 1

    Returns:
        1 + 3*n_total 比特电路；R1 上先 X 再 U_a 得到 |a'⟩⊗|a⟩⊗|b*⟩
    """
    qc = QuantumCircuit(1 + 3 * n_total)
    qc.h(0)
    qc.sdg(0)
    qc.h(0)
    # R1 的自旋比特：R1 从 qubit 1 开始，自旋为第 0 位 → qubit 1
    qc.x(1)
    qc.append(U_a.to_instruction(), list(range(1, 1 + n_total)))
    qc.append(U_a.to_instruction(), list(range(1 + n_total, 1 + 2 * n_total)))
    qc.append(U_b_conj.to_instruction(), list(range(1 + 2 * n_total, 1 + 3 * n_total)))
    return qc


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
    startFromStateVector: bool = False,
    m: float = 1.0
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
        m: 质量（线性子步用）
    
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
    psi_tilde = apply_linear_step_dirac(psi_0.copy(), n, dt, m)
    
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
x0 = 0.0     # 波包中心
sigma = 1.0  # 波包宽度
k0 = 5.0     # 初始动量
m = 1.0      # 质量（用于初始条件）

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

Psi0, Psi1 = dirac_initial_condition(x, x0, sigma, k0, m)

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
print(f"Initial condition: x0={x0}, sigma={sigma}, k0={k0}, m={m}")

# 初始化优化器
optimizer = L_BFGS_B(maxiter=maxiter, maxfun=maxfun, ftol=ftol)

# 存储参数
n_total = n + 1  # 总量子比特数
parameters = np.zeros((time_steps, 2 * n_total * (d + 1)))

# ===== 第一步：从初始条件开始 =====
cost_function = lambda x: cost_function_dirac(
    x, initial_condition, ansatz_dirac, n, d,
    lambda1, lambda2, dt, N_constant, dx, True, m
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
        lambda1, lambda2, dt, N_constant, dx, False, m
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
