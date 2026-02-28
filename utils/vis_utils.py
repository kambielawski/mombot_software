"""
Visualization utilities for Medea active learning.
"""

import os
import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if TYPE_CHECKING:
    from inverse import HeteroscedasticModel


def plot_active_learning_snapshot(
    model: "HeteroscedasticModel",
    X_train: np.ndarray,
    Y_train: np.ndarray,
    pre_velocity: float,
    selected_duration: float,
    predicted_post_v: float,
    output_path: str,
    pre_vals: np.ndarray = None,
    durations: np.ndarray = None,
    ucb_beta: float = 0.5,
    ucb_bandwidth: float = 20000,
    iteration: Optional[int] = None
):
    """
    Generate a 9-panel (3x3) snapshot of the active learning state.

    Parameters
    ----------
    model : HeteroscedasticModel
        Fitted model
    X_train : np.ndarray
        Training data, shape (n, 2) with columns [pre_val, duration]
    Y_train : np.ndarray
        Training labels (post_velocity)
    pre_velocity : float
        The current pre_velocity being evaluated
    selected_duration : float
        The duration selected by the acquisition function
    predicted_post_v : float
        The model's predicted post_velocity for the selected point
    output_path : str
        Path to save the output image
    pre_vals : np.ndarray, optional
        Pre-velocity values for heatmap grid (default: 0 to 1.6)
    durations : np.ndarray, optional
        Duration values for evaluation (default: 0 to 300000)
    ucb_beta : float
        UCB exploration weight
    ucb_bandwidth : float
        Bandwidth for density calculation
    iteration : int, optional
        Iteration number for title
    """
    if pre_vals is None:
        pre_vals = np.arange(0, 0.5, 0.01)
    if durations is None:
        durations = np.arange(0, 305000, 5000, dtype=float)

    # Compute variance heatmap
    variance_heat = np.empty((len(pre_vals), len(durations)))
    for i, pv in enumerate(pre_vals):
        X_query = np.column_stack([np.full(len(durations), pv), durations])
        variance_heat[i, :] = model.predict_variance(X_query)

    # Compute exploration bonus heatmap
    # Exploration bonus is based on duration density (same for all pre_vals)
    density = np.array([np.sum(np.abs(X_train[:, 1] - d) < ucb_bandwidth) for d in durations])
    exploration_bonus = 1.0 / (density + 1)
    exploration_normalized_full = exploration_bonus / (exploration_bonus.max() + 1e-8)

    # Create 2D exploration heatmap (tile across pre_vals since it only depends on duration)
    exploration_heat = np.tile(exploration_normalized_full, (len(pre_vals), 1))

    # Build full UCB acquisition heatmap
    ucb_heat = np.empty((len(pre_vals), len(durations)))
    for i, pv in enumerate(pre_vals):
        var_row = variance_heat[i, :]
        var_normalized = var_row / (var_row.max() + 1e-8)
        ucb_heat[i, :] = var_normalized + ucb_beta * exploration_normalized_full

    # Compute acquisition function for the selected pre_velocity slice
    slice_idx = np.argmin(np.abs(pre_vals - pre_velocity))
    variance_slice = variance_heat[slice_idx, :]
    variance_normalized = variance_slice / (variance_slice.max() + 1e-8)
    acquisition = variance_normalized + ucb_beta * exploration_normalized_full

    # Create figure with 3x3 layout
    fig = plt.figure(figsize=(18, 15))

    extent = [durations.min(), durations.max(), pre_vals.min(), pre_vals.max()]

    # ============ ROW 1: HEATMAPS ============

    # ---- Row 1, Plot 1: Variance Heatmap ----
    ax1 = fig.add_subplot(3, 3, 1)
    im1 = ax1.imshow(
        variance_heat,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='hot',
        vmin=variance_heat.min(),
        vmax=variance_heat.max()
    )
    ax1.set_xlabel('Duration (ms)')
    ax1.set_ylabel('Pre-estim velocity')
    ax1.set_title('Variance Heatmap')
    ax1.axhline(pre_velocity, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    ax1.scatter([selected_duration], [pre_velocity],
               c='cyan', s=120, marker='x', linewidths=3, zorder=10)
    fig.colorbar(im1, ax=ax1, label='Variance')

    # ---- Row 1, Plot 2: Exploration Bonus Heatmap ----
    ax2 = fig.add_subplot(3, 3, 2)
    im2 = ax2.imshow(
        exploration_heat,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='Greens',
        vmin=exploration_heat.min(),
        vmax=exploration_heat.max()
    )
    ax2.set_xlabel('Duration (ms)')
    ax2.set_ylabel('Pre-estim velocity')
    ax2.set_title(f'Exploration Bonus (bandwidth={ucb_bandwidth/1000:.0f}k)')
    ax2.axhline(pre_velocity, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    ax2.scatter([selected_duration], [pre_velocity],
               c='cyan', s=120, marker='x', linewidths=3, zorder=10)
    fig.colorbar(im2, ax=ax2, label='Exploration')

    # ---- Row 1, Plot 3: UCB Acquisition Heatmap ----
    ax3 = fig.add_subplot(3, 3, 3)
    im3 = ax3.imshow(
        ucb_heat,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='viridis',
        vmin=ucb_heat.min(),
        vmax=ucb_heat.max()
    )
    ax3.set_xlabel('Duration (ms)')
    ax3.set_ylabel('Pre-estim velocity')
    ax3.set_title(f'UCB Acquisition (beta={ucb_beta})')
    ax3.axhline(pre_velocity, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
    ax3.scatter([selected_duration], [pre_velocity],
               c='cyan', s=120, marker='x', linewidths=3, zorder=10)
    fig.colorbar(im3, ax=ax3, label='UCB')

    # ============ ROW 2: MARGINALS + SLICE ============

    # Compute marginal (expected) variance over pre_velocity
    marginal_variance = variance_heat.mean(axis=0)
    marginal_variance_normalized = marginal_variance / (marginal_variance.max() + 1e-8)

    # Compute marginal (expected) acquisition over pre_velocity
    marginal_acquisition = ucb_heat.mean(axis=0)

    # ---- Row 2, Plot 4: Marginal Expected Variance ----
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(durations, marginal_variance, 'r-', linewidth=2, label='E[Variance]')
    ax4.axvline(selected_duration, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Duration (ms)')
    ax4.set_ylabel('Expected Variance')
    ax4.set_title('Marginal Variance (mean over pre_vel)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=7)

    # ---- Row 2, Plot 5: UCB Acquisition Slice at selected pre_velocity ----
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(durations, variance_normalized, 'b-', linewidth=2, label='Variance (norm)', alpha=0.7)
    ax5.plot(durations, exploration_normalized_full, 'g-', linewidth=2, label='Exploration', alpha=0.7)
    ax5.plot(durations, acquisition, 'k-', linewidth=2.5, label='UCB')
    ax5.axvline(selected_duration, color='red', linestyle='--', linewidth=2)
    best_idx = np.argmin(np.abs(durations - selected_duration))
    ax5.scatter([selected_duration], [acquisition[best_idx]],
               c='red', s=100, zorder=10, label=f'Selected: {selected_duration:.0f}')
    ax5.set_title(f'UCB Slice at pre_val={pre_velocity:.2f}')
    ax5.legend(loc='upper right', fontsize=7)
    ax5.set_xlabel('Duration (ms)')
    ax5.set_ylabel('Acquisition value')
    ax5.set_ylim(0, 2.0)
    ax5.grid(True, alpha=0.3)

    # ---- Row 2, Plot 6: Marginal Expected Acquisition ----
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(durations, marginal_acquisition, 'purple', linewidth=2, label='E[UCB]')
    ax6.axvline(selected_duration, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax6.set_xlabel('Duration (ms)')
    ax6.set_ylabel('Expected Acquisition')
    ax6.set_title('Marginal Acquisition (mean over pre_vel)')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper right', fontsize=7)

    # ============ ROW 3: SCATTER PLOTS + 3D ============

    # ---- Row 3, Plot 7: Pre-velocity vs Duration scatter ----
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.scatter(
        X_train[:, 1], X_train[:, 0],
        c=Y_train, cmap='coolwarm',
        s=60, edgecolors='black', linewidths=0.5,
        vmin=0, vmax=1.5,
        label='Training data'
    )
    ax7.scatter(
        [selected_duration], [pre_velocity],
        c='lime', s=200, marker='*',
        edgecolors='black', linewidths=2, zorder=10,
        label=f'Selected: ({selected_duration:.0f}, {pre_velocity:.2f})'
    )
    ax7.set_xlabel('Duration (ms)')
    ax7.set_ylabel('Pre-estim velocity')
    ax7.set_title(f'Pre-velocity vs Duration ({len(X_train)} pts)')
    ax7.set_xlim(durations.min(), durations.max()+1)
    ax7.set_ylim(0, 1.6)
    ax7.legend(loc='upper right', fontsize=7)

    # ---- Row 3, Plot 8: Post-velocity vs Duration scatter ----
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.scatter(
        X_train[:, 1], Y_train,
        c=Y_train, cmap='coolwarm',
        s=60, edgecolors='black', linewidths=0.5,
        vmin=0, vmax=1.5,
        label='Training data'
    )
    ax8.scatter(
        [selected_duration], [predicted_post_v],
        c='lime', s=200, marker='*',
        edgecolors='black', linewidths=2, zorder=10,
        label=f'Predicted: {predicted_post_v:.2f}'
    )
    ax8.set_xlabel('Duration (ms)')
    ax8.set_ylabel('Post-estim velocity')
    ax8.set_title(f'Post-velocity vs Duration ({len(X_train)} pts)')
    ax8.set_xlim(durations.min(), durations.max()+1)
    ax8.set_ylim(0, 1.6)
    ax8.legend(loc='upper right', fontsize=7)

    # ---- Row 3, Plot 9: 3D surface with linear regression model ----
    ax9 = fig.add_subplot(3, 3, 9, projection='3d')
    dur_grid = np.linspace(durations.min(), durations.max(), 50)
    pre_grid = np.linspace(0, 1.5, 50)
    DUR, PRE = np.meshgrid(dur_grid, pre_grid)
    X_surface = np.column_stack([PRE.ravel(), DUR.ravel()])
    Z_surface = model.predict_mean(X_surface).reshape(DUR.shape)
    ax9.plot_surface(DUR, PRE, Z_surface, cmap='coolwarm', alpha=0.6, edgecolor='none')
    ax9.scatter(
        X_train[:, 1], X_train[:, 0], Y_train,
        c=Y_train, cmap='coolwarm', s=60, edgecolors='black', linewidths=0.5,
        vmin=0, vmax=1.5, depthshade=False
    )
    ax9.scatter(
        [selected_duration], [pre_velocity], [predicted_post_v],
        c='lime', s=200, marker='*', edgecolors='black', linewidths=2,
        depthshade=False
    )
    ax9.set_xlabel('Duration (ms)')
    ax9.set_ylabel('Pre-estim vel')
    ax9.set_zlabel('Post-estim vel')
    ax9.set_title('3D Model Surface')

    # Title
    title = 'Active Learning Snapshot'
    if iteration is not None:
        title += f' - Iteration {iteration}'
    fig.suptitle(title, fontsize=14, y=0.995)

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved active learning snapshot to {output_path}")
