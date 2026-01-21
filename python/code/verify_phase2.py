"""
Phase 2 verification script - generates figures to validate profile extraction.
Run from python directory: python code/verify_phase2.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import profiles


def create_synthetic_beach_3d():
    """
    Create synthetic 3D beach topography with:
    - Sloped beach surface with alongshore variation
    - Some noise and outliers
    """
    np.random.seed(42)
    n_points = 5000

    # Points spread across beach
    X = np.random.uniform(0, 100, n_points)  # Cross-shore (0=backshore, 100=offshore)
    Y = np.random.uniform(-20, 20, n_points)  # Alongshore

    # Beach profile: z = -0.08*x + 0.005*y + noise
    # Slopes down offshore, slight alongshore variation
    Z = -0.08 * X + 0.005 * Y + 2.0  # 2m at backshore
    Z += np.random.normal(0, 0.05, n_points)  # Small noise

    # Add some outliers (vegetation/debris)
    n_outliers = int(n_points * 0.05)
    outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
    Z[outlier_idx] += np.random.uniform(0.5, 2.0, n_outliers)

    return X, Y, Z


def fig1_transect_geometry():
    """Figure 1: Show transect geometry and point selection"""
    X, Y, Z = create_synthetic_beach_3d()

    # Define transects
    config = profiles.TransectConfig(
        x1=0, y1=0,
        x2=100, y2=0,
        alongshore_spacings=(-15, -10, -5, 0, 5, 10, 15),
        resolution=1.0,
        tolerance=2.0,
        extend_line=(0, 0),
        outlier_threshold=0.4,
        max_gap=5.0,
    )

    result = profiles.extract_transects(X, Y, Z, config)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Plan view with transect lines
    ax = axes[0]
    sc = ax.scatter(X, Y, c=Z, s=1, cmap='terrain', vmin=-8, vmax=4, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Elevation (m)')

    # Draw transect lines
    colors = plt.cm.tab10(np.linspace(0, 1, len(config.alongshore_spacings)))
    for i, (xt, yt) in enumerate(result.transect_coords):
        ax.plot(xt, yt, '-', color=colors[i], linewidth=2,
                label=f'Y={config.alongshore_spacings[i]}m')

    ax.set_xlabel('X (cross-shore, m)')
    ax.set_ylabel('Y (alongshore, m)')
    ax.set_title('Plan View with Transect Lines')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')

    # Right: All extracted profiles
    ax = axes[1]
    for i, offset in enumerate(config.alongshore_spacings):
        z_profile = result.Z3D[i, :]
        valid = ~np.isnan(z_profile)
        ax.plot(result.x1d[valid], z_profile[valid], '-', color=colors[i],
                label=f'Y={offset}m', linewidth=1.5)

    # True profile at y=0
    x_true = np.linspace(0, 100, 100)
    z_true = -0.08 * x_true + 2.0
    ax.plot(x_true, z_true, 'k--', linewidth=2, label='True (Y=0)')

    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Extracted Profiles')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 1: Transect Geometry and Profile Extraction', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def fig2_outlier_removal():
    """Figure 2: Show quadratic outlier removal effect"""
    np.random.seed(42)

    # Create 1D profile with outliers
    x = np.linspace(0, 50, 200)
    z_true = -0.1 * x + 2.0  # Linear beach profile
    z_noisy = z_true + np.random.normal(0, 0.03, len(x))

    # Add some outliers
    outlier_idx = [30, 50, 80, 120, 150]
    z_with_outliers = z_noisy.copy()
    for idx in outlier_idx:
        z_with_outliers[idx] += np.random.uniform(0.5, 1.5)

    # Apply quadratic outlier removal
    z_clean = profiles._fit_quadratic_and_remove_outliers(
        x, z_with_outliers, threshold=0.4
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Before cleaning
    ax = axes[0]
    ax.scatter(x, z_with_outliers, c='gray', s=10, alpha=0.7, label='All points')
    ax.scatter(x[outlier_idx], z_with_outliers[outlier_idx], c='red', s=30, label='Outliers')
    ax.plot(x, z_true, 'k-', linewidth=2, label='True surface')
    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Before Outlier Removal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Quadratic fit visualization
    ax = axes[1]
    valid = ~np.isnan(z_clean)
    coeffs = np.polyfit(x[valid], z_clean[valid], 2)
    z_fit = np.polyval(coeffs, x)

    ax.scatter(x, z_with_outliers, c='gray', s=10, alpha=0.5)
    ax.plot(x, z_fit, 'b-', linewidth=2, label='Quadratic fit')
    ax.fill_between(x, z_fit - 0.4, z_fit + 0.4, alpha=0.2, color='blue',
                    label='Threshold (±0.4m)')
    ax.scatter(x[~valid], z_with_outliers[~valid], c='red', s=30, marker='x',
               label='Removed')
    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Quadratic Fit and Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # After cleaning
    ax = axes[2]
    ax.scatter(x[valid], z_clean[valid], c='green', s=10, alpha=0.7, label='Cleaned points')
    ax.plot(x, z_true, 'k-', linewidth=2, label='True surface')
    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('After Outlier Removal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 2: Quadratic Outlier Removal', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def fig3_gap_interpolation():
    """Figure 3: Show gap interpolation behavior"""
    x = np.arange(0, 50, 0.5)
    z_true = -0.1 * x + 2.0

    # Create profile with gaps
    z_gapped = z_true.copy()

    # Small gap (2m) - should be filled
    z_gapped[20:24] = np.nan  # 2m gap

    # Large gap (8m) - should NOT be filled
    z_gapped[60:76] = np.nan  # 8m gap

    # Compute gap sizes
    gaps = profiles.gapsize(z_gapped)

    # Apply inpaint_nans
    z_filled = profiles.inpaint_nans(x, z_gapped, max_gap=4.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original with gaps
    ax = axes[0]
    valid = ~np.isnan(z_gapped)
    ax.plot(x[valid], z_gapped[valid], 'b.-', markersize=4, label='Data with gaps')
    ax.plot(x, z_true, 'k--', alpha=0.5, label='True surface')

    # Highlight gaps
    gap_mask = np.isnan(z_gapped)
    ax.axvspan(x[20], x[23], alpha=0.3, color='green', label='Small gap (2m)')
    ax.axvspan(x[60], x[75], alpha=0.3, color='red', label='Large gap (8m)')

    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Profile with Gaps')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Gap sizes
    ax = axes[1]
    ax.bar(x, gaps * 0.5, width=0.4, color='orange', alpha=0.7)  # Convert to meters
    ax.axhline(y=4.0, color='red', linestyle='--', label='Max gap threshold (4m)')
    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Gap size (m)')
    ax.set_title('Gap Sizes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # After interpolation
    ax = axes[2]
    valid_filled = ~np.isnan(z_filled)
    ax.plot(x, z_true, 'k--', alpha=0.5, label='True surface')
    ax.plot(x[valid_filled], z_filled[valid_filled], 'g.-', markersize=4, label='After interpolation')

    # Show where large gap remains
    still_nan = np.isnan(z_filled)
    if still_nan.any():
        ax.scatter(x[still_nan], np.zeros(still_nan.sum()) - 3, c='red', s=20,
                   marker='x', label='Still NaN (large gap)')

    ax.set_xlabel('Cross-shore distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('After Gap Interpolation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 3: Gap Interpolation (max_gap=4m)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def fig4_3d_visualization():
    """Figure 4: 3D visualization of extracted profiles"""
    X, Y, Z = create_synthetic_beach_3d()

    config = profiles.TransectConfig(
        x1=0, y1=0,
        x2=100, y2=0,
        alongshore_spacings=(-15, -10, -5, 0, 5, 10, 15),
        resolution=0.5,
        tolerance=3.0,
        extend_line=(0, 0),
    )

    result = profiles.extract_transects(X, Y, Z, config)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot raw point cloud (subsampled)
    subsample = np.random.choice(len(X), min(2000, len(X)), replace=False)
    ax.scatter(X[subsample], Y[subsample], Z[subsample], c='gray', s=1, alpha=0.3, label='Point cloud')

    # Plot extracted profiles
    colors = plt.cm.viridis(np.linspace(0, 1, len(config.alongshore_spacings)))
    for i, offset in enumerate(config.alongshore_spacings):
        xt, yt = result.transect_coords[i]
        z_profile = result.Z3D[i, :]
        valid = ~np.isnan(z_profile)

        # Map x1d to UTM coordinates
        x_utm, y_utm = profiles.transect_to_utm(
            result.x1d, config.x1, config.y1, config.x2, config.y2,
            extend_line=config.extend_line
        )
        # Shift to correct alongshore position
        y_utm = y_utm + offset

        ax.plot(x_utm[valid], y_utm[valid], z_profile[valid],
                color=colors[i], linewidth=2, label=f'Y={offset}m')

    ax.set_xlabel('X (cross-shore)')
    ax.set_ylabel('Y (alongshore)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D View: Point Cloud and Extracted Profiles')
    ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Figure 4: 3D Profile Extraction Visualization', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    output_dir = Path(__file__).parent.parent / 'figures' / 'tests'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Phase 2 verification figures...")

    fig1 = fig1_transect_geometry()
    fig1.savefig(output_dir / 'fig5_transect_geometry.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig5_transect_geometry.png'}")

    fig2 = fig2_outlier_removal()
    fig2.savefig(output_dir / 'fig6_outlier_removal.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig6_outlier_removal.png'}")

    fig3 = fig3_gap_interpolation()
    fig3.savefig(output_dir / 'fig7_gap_interpolation.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig7_gap_interpolation.png'}")

    fig4 = fig4_3d_visualization()
    fig4.savefig(output_dir / 'fig8_3d_profiles.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig8_3d_profiles.png'}")

    print(f"\nAll figures saved to: {output_dir}")
    print("\nFigure descriptions:")
    print("  5. Transect geometry and profile extraction")
    print("  6. Quadratic outlier removal process")
    print("  7. Gap interpolation behavior")
    print("  8. 3D visualization of extracted profiles")

    plt.show()


if __name__ == '__main__':
    main()
