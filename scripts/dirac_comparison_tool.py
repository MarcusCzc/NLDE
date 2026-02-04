"""
对比分析工具：比较VQA、经典和归一化经典三种方法的结果
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from numpy.typing import NDArray


def calculate_rmse_dirac(
    Psi0_num: NDArray,
    Psi1_num: NDArray,
    Psi0_ref: NDArray,
    Psi1_ref: NDArray
) -> float:
    """
    计算Dirac方程的RMSE（对应公式31）
    
    RMSE = sqrt( 1/(2^n) * sum_{σ,j} (|Ψ^num_{σ,j}| - |Ψ^ref_{σ,j}|)² )
    
    Args:
        Psi0_num: 数值解的上自旋分量
        Psi1_num: 数值解的下自旋分量
        Psi0_ref: 参考解的上自旋分量
        Psi1_ref: 参考解的下自旋分量
    
    Returns:
        RMSE值
    """
    N = len(Psi0_num)
    
    # 计算每个分量的模长差异
    diff0 = np.abs(Psi0_num) - np.abs(Psi0_ref)
    diff1 = np.abs(Psi1_num) - np.abs(Psi1_ref)
    
    # 计算RMSE
    rmse = np.sqrt((np.sum(diff0**2) + np.sum(diff1**2)) / (2 * N))
    
    return rmse


def calculate_l2_error_dirac(
    Psi0_num: NDArray,
    Psi1_num: NDArray,
    Psi0_ref: NDArray,
    Psi1_ref: NDArray
) -> float:
    """
    计算L2误差（对应公式32）
    
    L2_error = sqrt( 1/(2^n) * sum_j (‖Ψ^num_j - Ψ^ref_j‖²) )
    
    其中 Ψ_j = (Ψ0_j, Ψ1_j) 是双分量向量
    """
    N = len(Psi0_num)
    
    # 计算每个空间点的双分量向量差异
    diff0 = Psi0_num - Psi0_ref
    diff1 = Psi1_num - Psi1_ref
    
    # 计算L2范数
    l2_error = np.sqrt(np.sum(np.abs(diff0)**2 + np.abs(diff1)**2) / N)
    
    return l2_error


def calculate_conservation_errors(
    Psi0_history: NDArray,
    Psi1_history: NDArray,
    dx: float,
    N_constant: float = 1.0
) -> Tuple[NDArray, NDArray]:
    """
    计算守恒量误差
    
    Returns:
        (norm_errors, spin_pol_evolution): 归一化误差和自旋极化演化
    """
    num_steps = len(Psi0_history)
    norm_errors = np.zeros(num_steps)
    spin_pol_total = np.zeros(num_steps)
    
    for i in range(num_steps):
        # 计算归一化
        total_norm = np.sum(np.abs(Psi0_history[i])**2 + np.abs(Psi1_history[i])**2) * dx
        norm_errors[i] = abs(total_norm - N_constant)
        
        # 计算总自旋极化
        spin_pol = np.sum(np.abs(Psi0_history[i])**2 - np.abs(Psi1_history[i])**2) * dx
        spin_pol_total[i] = spin_pol
    
    return norm_errors, spin_pol_total


def compare_methods(
    vqa_data_path: str,
    classical_data_path: str,
    normalized_data_path: str,
    output_dir: str = "/home/claude"
):
    """
    比较三种方法的结果
    
    Args:
        vqa_data_path: VQA结果数据路径
        classical_data_path: 经典方法结果数据路径
        normalized_data_path: 归一化经典方法结果数据路径
        output_dir: 输出目录
    """
    # 加载数据
    print("Loading data...")
    vqa_data = np.load(vqa_data_path, allow_pickle=True)
    classical_data = np.load(classical_data_path, allow_pickle=True)
    normalized_data = np.load(normalized_data_path, allow_pickle=True)
    
    # 提取波函数（VQA的数据格式不同，需要分离）
    vqa_psi = vqa_data['psi']
    N = vqa_psi.shape[1] // 2
    vqa_Psi0 = vqa_psi[:, 0:N]
    vqa_Psi1 = vqa_psi[:, N:2*N]
    
    classical_Psi0 = classical_data['Psi0']
    classical_Psi1 = classical_data['Psi1']
    
    normalized_Psi0 = normalized_data['Psi0']
    normalized_Psi1 = normalized_data['Psi1']
    
    x = classical_data['x']
    t = classical_data['t']
    dx = x[1] - x[0]
    
    # 使用归一化经典解作为参考
    ref_Psi0 = normalized_Psi0
    ref_Psi1 = normalized_Psi1
    
    # ========== 计算RMSE随时间的变化 ==========
    print("Calculating RMSE...")
    num_steps = len(t)
    
    vqa_rmse = np.zeros(num_steps)
    classical_rmse = np.zeros(num_steps)
    
    for i in range(num_steps):
        vqa_rmse[i] = calculate_rmse_dirac(
            vqa_Psi0[i], vqa_Psi1[i], ref_Psi0[i], ref_Psi1[i]
        )
        classical_rmse[i] = calculate_rmse_dirac(
            classical_Psi0[i], classical_Psi1[i], ref_Psi0[i], ref_Psi1[i]
        )
    
    # ========== 计算守恒量误差 ==========
    print("Calculating conservation errors...")
    vqa_norm_err, vqa_spin_pol = calculate_conservation_errors(
        vqa_Psi0, vqa_Psi1, dx
    )
    classical_norm_err, classical_spin_pol = calculate_conservation_errors(
        classical_Psi0, classical_Psi1, dx
    )
    normalized_norm_err, normalized_spin_pol = calculate_conservation_errors(
        normalized_Psi0, normalized_Psi1, dx
    )
    
    # ========== 可视化对比 ==========
    print("Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 第一行：RMSE对比
    ax = axes[0, 0]
    ax.semilogy(t, vqa_rmse, 'o-', label='VQA', markersize=4)
    ax.semilogy(t, classical_rmse, 's-', label='Classical', markersize=4)
    ax.set_xlabel('Time')
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Square Error vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 归一化误差对比
    ax = axes[0, 1]
    ax.semilogy(t, vqa_norm_err, 'o-', label='VQA', markersize=4)
    ax.semilogy(t, classical_norm_err, 's-', label='Classical', markersize=4)
    ax.semilogy(t, normalized_norm_err, '^-', label='Normalized Classical', markersize=4)
    ax.set_xlabel('Time')
    ax.set_ylabel('|Norm - 1|')
    ax.set_title('Normalization Error vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 自旋极化演化
    ax = axes[0, 2]
    ax.plot(t, vqa_spin_pol, 'o-', label='VQA', markersize=4)
    ax.plot(t, classical_spin_pol, 's-', label='Classical', markersize=4)
    ax.plot(t, normalized_spin_pol, '^-', label='Normalized Classical', markersize=4)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Spin Polarization')
    ax.set_title('Spin Polarization vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 第二行：最后时刻的波函数对比
    final_idx = -1
    
    # 上自旋分量对比
    ax = axes[1, 0]
    ax.plot(x/np.pi, np.abs(ref_Psi0[final_idx])**2, 'k-', linewidth=2, label='Reference')
    ax.plot(x/np.pi, np.abs(vqa_Psi0[final_idx])**2, 'o-', markersize=4, label='VQA')
    ax.plot(x/np.pi, np.abs(classical_Psi0[final_idx])**2, 's-', markersize=4, label='Classical')
    ax.set_xlabel('$x$ [$\\pi$]')
    ax.set_ylabel('$|\\Psi_0|^2$')
    ax.set_title(f'Spin-up at t={t[final_idx]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 下自旋分量对比
    ax = axes[1, 1]
    ax.plot(x/np.pi, np.abs(ref_Psi1[final_idx])**2, 'k-', linewidth=2, label='Reference')
    ax.plot(x/np.pi, np.abs(vqa_Psi1[final_idx])**2, 'o-', markersize=4, label='VQA')
    ax.plot(x/np.pi, np.abs(classical_Psi1[final_idx])**2, 's-', markersize=4, label='Classical')
    ax.set_xlabel('$x$ [$\\pi$]')
    ax.set_ylabel('$|\\Psi_1|^2$')
    ax.set_title(f'Spin-down at t={t[final_idx]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 总概率密度对比
    ax = axes[1, 2]
    ref_total = np.abs(ref_Psi0[final_idx])**2 + np.abs(ref_Psi1[final_idx])**2
    vqa_total = np.abs(vqa_Psi0[final_idx])**2 + np.abs(vqa_Psi1[final_idx])**2
    classical_total = np.abs(classical_Psi0[final_idx])**2 + np.abs(classical_Psi1[final_idx])**2
    
    ax.plot(x/np.pi, ref_total, 'k-', linewidth=2, label='Reference')
    ax.plot(x/np.pi, vqa_total, 'o-', markersize=4, label='VQA')
    ax.plot(x/np.pi, classical_total, 's-', markersize=4, label='Classical')
    ax.set_xlabel('$x$ [$\\pi$]')
    ax.set_ylabel('Total $|\\Psi|^2$')
    ax.set_title(f'Total Probability at t={t[final_idx]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/dirac_comparison_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    
    # ========== 打印统计信息 ==========
    print("\n" + "="*60)
    print("COMPARISON STATISTICS")
    print("="*60)
    print(f"\nFinal RMSE (at t={t[-1]:.2f}):")
    print(f"  VQA:       {vqa_rmse[-1]:.6e}")
    print(f"  Classical: {classical_rmse[-1]:.6e}")
    
    print(f"\nMean RMSE (over all time):")
    print(f"  VQA:       {np.mean(vqa_rmse):.6e}")
    print(f"  Classical: {np.mean(classical_rmse):.6e}")
    
    print(f"\nMax Normalization Error:")
    print(f"  VQA:                  {np.max(vqa_norm_err):.6e}")
    print(f"  Classical:            {np.max(classical_norm_err):.6e}")
    print(f"  Normalized Classical: {np.max(normalized_norm_err):.6e}")
    
    print(f"\nSpin Polarization Change:")
    print(f"  VQA:                  {abs(vqa_spin_pol[-1] - vqa_spin_pol[0]):.6e}")
    print(f"  Classical:            {abs(classical_spin_pol[-1] - classical_spin_pol[0]):.6e}")
    print(f"  Normalized Classical: {abs(normalized_spin_pol[-1] - normalized_spin_pol[0]):.6e}")
    
    plt.show()


if __name__ == "__main__":
    # 示例用法
    compare_methods(
        vqa_data_path="/home/claude/dirac_vqa_data.npz",
        classical_data_path="/home/claude/dirac_classical_data.npz",
        normalized_data_path="/home/claude/dirac_normalized_classical_data.npz",
        output_dir="/home/claude"
    )
