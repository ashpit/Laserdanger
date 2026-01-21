"""
Phase 1 verification script - generates figures to validate algorithm outputs.
Run from python directory: python code/verify_phase1.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import phase2


def create_synthetic_beach():
    """
    Create synthetic beach topography with:
    - Sloped beach surface
    - Some noise
    - Vegetation/outlier points above surface
    """
    np.random.seed(42)
    n_ground = 2000
    n_vegetation = 200

    # Ground points on sloped beach: z = -0.1*x + 0.02*y + noise
    x_ground = np.random.uniform(0, 50, n_ground)
    y_ground = np.random.uniform(0, 30, n_ground)
    z_ground = -0.1 * x_ground + 0.02 * y_ground + np.random.normal(0, 0.03, n_ground)

    # Vegetation/noise points 0.5-2m above ground
    x_veg = np.random.uniform(0, 50, n_vegetation)
    y_veg = np.random.uniform(0, 30, n_vegetation)
    z_veg = -0.1 * x_veg + 0.02 * y_veg + np.random.uniform(0.5, 2.0, n_vegetation)

    ground = np.column_stack([x_ground, y_ground, z_ground])
    vegetation = np.column_stack([x_veg, y_veg, z_veg])

    return ground, vegetation


def fig1_binned_grid():
    """Figure 1: Binned grid DEM visualization"""
    ground, vegetation = create_synthetic_beach()
    pts = np.vstack([ground, vegetation])

    # Bin without percentile filter (includes vegetation)
    grid_all = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=None)

    # Bin with 50th percentile filter (ground surface)
    grid_filtered = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=50)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Raw point cloud
    ax = axes[0]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=1, cmap='terrain', vmin=-6, vmax=2)
    ax.set_xlabel('X (cross-shore)')
    ax.set_ylabel('Y (alongshore)')
    ax.set_title('Raw Point Cloud')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='Z (m)')

    # Binned without filter
    ax = axes[1]
    x_centers = (grid_all.x_edges[:-1] + grid_all.x_edges[1:]) / 2
    y_centers = (grid_all.y_edges[:-1] + grid_all.y_edges[1:]) / 2
    im = ax.pcolormesh(x_centers, y_centers, grid_all.z_mean.T, cmap='terrain', vmin=-6, vmax=2)
    ax.set_xlabel('X (cross-shore)')
    ax.set_ylabel('Y (alongshore)')
    ax.set_title('Binned (No Percentile Filter)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Z mean (m)')

    # Binned with 50th percentile filter
    ax = axes[2]
    im = ax.pcolormesh(x_centers, y_centers, grid_filtered.z_mean.T, cmap='terrain', vmin=-6, vmax=2)
    ax.set_xlabel('X (cross-shore)')
    ax.set_ylabel('Y (alongshore)')
    ax.set_title('Binned (50th Percentile Filter)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Z mean (m)')

    fig.suptitle('Figure 1: Percentile Filtering Effect on Binned Grid', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def fig2_snr_mask():
    """Figure 2: SNR mask visualization"""
    np.random.seed(42)

    # Create points with varying density and noise
    # Dense, low noise region (high SNR)
    n1 = 500
    x1 = np.random.uniform(0, 15, n1)
    y1 = np.random.uniform(0, 15, n1)
    z1 = np.random.normal(5.0, 0.02, n1)  # Very low std
    pts1 = np.column_stack([x1, y1, z1])

    # Sparse region (low count)
    n2 = 20
    x2 = np.random.uniform(20, 35, n2)
    y2 = np.random.uniform(0, 15, n2)
    z2 = np.random.normal(5.0, 0.02, n2)
    pts2 = np.column_stack([x2, y2, z2])

    # High noise region (low SNR)
    n3 = 300
    x3 = np.random.uniform(0, 15, n3)
    y3 = np.random.uniform(20, 35, n3)
    z3 = np.random.normal(5.0, 1.5, n3)  # High std
    pts3 = np.column_stack([x3, y3, z3])

    pts = np.vstack([pts1, pts2, pts3])

    grid = phase2.bin_point_cloud(pts, bin_size=5.0, percentile=None)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    x_centers = (grid.x_edges[:-1] + grid.x_edges[1:]) / 2
    y_centers = (grid.y_edges[:-1] + grid.y_edges[1:]) / 2

    # Point count
    ax = axes[0]
    im = ax.pcolormesh(x_centers, y_centers, grid.count.T, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Point Count per Bin')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Count')

    # Standard deviation
    ax = axes[1]
    im = ax.pcolormesh(x_centers, y_centers, grid.z_std.T, cmap='plasma', vmin=0, vmax=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Z Standard Deviation')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Std (m)')

    # SNR
    ax = axes[2]
    snr_display = np.clip(grid.snr, 0, 500)
    im = ax.pcolormesh(x_centers, y_centers, snr_display.T, cmap='RdYlGn', vmin=0, vmax=500)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('SNR (clipped to 500)')
    ax.set_aspect('equal')
    ax.axhline(y=15, color='white', linestyle='--', alpha=0.5)
    ax.axvline(x=15, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(im, ax=ax, label='SNR')

    # Valid mask
    ax = axes[3]
    im = ax.pcolormesh(x_centers, y_centers, grid.valid_mask.T.astype(int), cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Valid Mask (SNR>=100 & count>10)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Valid', ticks=[0, 1])

    fig.suptitle('Figure 2: SNR-Based Quality Filtering', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def fig3_residual_filter():
    """Figure 3: Residual kernel filtering - before and after"""
    ground, vegetation = create_synthetic_beach()
    pts = np.vstack([ground, vegetation])

    # Apply two-stage filter
    filtered_pts = phase2.residual_kernel_filter_two_stage(
        pts,
        passes=[(10.0, 0.2), (3.0, 0.1)],
        min_points_per_cell=10
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Before filtering - side view
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 2], c='gray', s=1, alpha=0.5, label='All points')
    ax.set_xlabel('X (cross-shore)')
    ax.set_ylabel('Z (elevation)')
    ax.set_title(f'Before Filtering (n={len(pts)})')
    ax.set_ylim(-7, 3)
    ax.legend()

    # After filtering - side view
    ax = axes[1]
    ax.scatter(filtered_pts[:, 0], filtered_pts[:, 2], c='green', s=1, alpha=0.5, label='Ground points')
    ax.set_xlabel('X (cross-shore)')
    ax.set_ylabel('Z (elevation)')
    ax.set_title(f'After Two-Stage Filter (n={len(filtered_pts)})')
    ax.set_ylim(-7, 3)
    ax.legend()

    # Comparison overlay
    ax = axes[2]
    # Plot removed points in red
    removed_mask = np.ones(len(pts), dtype=bool)
    # Find which original points are approximately in filtered (by matching coordinates)
    for fp in filtered_pts:
        dists = np.linalg.norm(pts - fp, axis=1)
        closest = np.argmin(dists)
        if dists[closest] < 0.01:
            removed_mask[closest] = False

    removed_pts = pts[removed_mask]
    kept_pts = pts[~removed_mask]

    ax.scatter(removed_pts[:, 0], removed_pts[:, 2], c='red', s=2, alpha=0.5, label=f'Removed (n={len(removed_pts)})')
    ax.scatter(kept_pts[:, 0], kept_pts[:, 2], c='green', s=2, alpha=0.5, label=f'Kept (n={len(kept_pts)})')
    ax.set_xlabel('X (cross-shore)')
    ax.set_ylabel('Z (elevation)')
    ax.set_title('Filtering Comparison')
    ax.set_ylim(-7, 3)
    ax.legend()

    fig.suptitle('Figure 3: Two-Stage Residual Kernel Filtering', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def fig4_cross_section():
    """Figure 4: Cross-section comparison showing ground extraction"""
    ground, vegetation = create_synthetic_beach()
    pts = np.vstack([ground, vegetation])

    # Bin with different settings
    grid_raw = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=None)
    grid_p50 = phase2.bin_point_cloud(pts, bin_size=1.0, percentile=50)

    x_centers = (grid_raw.x_edges[:-1] + grid_raw.x_edges[1:]) / 2

    # Extract a cross-section at y ~ 15m
    y_centers = (grid_raw.y_edges[:-1] + grid_raw.y_edges[1:]) / 2
    y_idx = np.argmin(np.abs(y_centers - 15))

    fig, ax = plt.subplots(figsize=(12, 5))

    # True ground surface
    x_true = np.linspace(0, 50, 100)
    z_true = -0.1 * x_true + 0.02 * 15  # At y=15

    ax.plot(x_true, z_true, 'k-', linewidth=2, label='True Ground Surface')
    ax.plot(x_centers, grid_raw.z_mean[:, y_idx], 'r--', linewidth=1.5, label='Binned (no filter) - mean')
    ax.plot(x_centers, grid_p50.z_mean[:, y_idx], 'g-', linewidth=1.5, label='Binned (50th percentile) - mean')
    ax.plot(x_centers, grid_raw.z_min[:, y_idx], 'b:', linewidth=1, label='Binned (no filter) - min')

    ax.set_xlabel('X (cross-shore distance, m)')
    ax.set_ylabel('Z (elevation, m)')
    ax.set_title(f'Cross-Section at Y = {y_centers[y_idx]:.1f}m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 4: Cross-Shore Profile Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    output_dir = Path(__file__).parent.parent / 'figures' / 'tests'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Phase 1 verification figures...")

    # Generate all figures
    fig1 = fig1_binned_grid()
    fig1.savefig(output_dir / 'fig1_binned_grid.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig1_binned_grid.png'}")

    fig2 = fig2_snr_mask()
    fig2.savefig(output_dir / 'fig2_snr_mask.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig2_snr_mask.png'}")

    fig3 = fig3_residual_filter()
    fig3.savefig(output_dir / 'fig3_residual_filter.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig3_residual_filter.png'}")

    fig4 = fig4_cross_section()
    fig4.savefig(output_dir / 'fig4_cross_section.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig4_cross_section.png'}")

    print(f"\nAll figures saved to: {output_dir}")
    print("\nFigure descriptions:")
    print("  1. Binned grid showing percentile filtering effect")
    print("  2. SNR mask showing quality-based filtering")
    print("  3. Residual kernel filtering before/after comparison")
    print("  4. Cross-section profile comparison")

    plt.show()


if __name__ == '__main__':
    main()
