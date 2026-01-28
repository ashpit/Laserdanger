#!/usr/bin/env python3
"""
Generate a visual diagram of the Laserdanger L1/L2 processing pipeline.

Creates a publication-quality figure showing the data flow from raw LAZ files
through L1 (surfaces) and L2 (timestacks) processing.

Usage:
    python scripts/visualization/create_pipeline_diagram.py [--output pipeline_diagram.png]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_pipeline_diagram(output_path: Path, dpi: int = 150):
    """Create the pipeline overview diagram."""

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Colors
    colors = {
        'input': '#E8F4FD',      # Light blue
        'process': '#FFF3E0',    # Light orange
        'output': '#E8F5E9',     # Light green
        'arrow': '#37474F',      # Dark gray
        'text': '#212121',       # Almost black
        'l1': '#2196F3',         # Blue
        'l2': '#FF9800',         # Orange
        'header': '#1565C0',     # Dark blue
    }

    def draw_box(x, y, w, h, text, color, fontsize=10, bold=False):
        """Draw a rounded box with text."""
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.2",
            facecolor=color,
            edgecolor='#424242',
            linewidth=1.5,
        )
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=colors['text'],
                wrap=True)
        return box

    def draw_arrow(x1, y1, x2, y2, color='#37474F'):
        """Draw an arrow between points."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Title
    ax.text(8, 11.5, 'Laserdanger LiDAR Processing Pipeline',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=colors['header'])
    ax.text(8, 11.0, 'Livox Avia Point Cloud → Beach Surfaces & Wave Dynamics',
            ha='center', va='center', fontsize=11, color='#616161')

    # =========== INPUT SECTION ===========
    # Scanner box
    draw_box(6.5, 9.2, 3, 0.8, 'Livox Avia Scanner\n(on tower)', colors['input'], fontsize=10, bold=True)

    # LAZ files
    draw_box(6.5, 8.0, 3, 0.8, '.laz files\n(timestamped, ~30s each)', colors['input'], fontsize=9)

    draw_arrow(8, 9.2, 8, 8.85)

    # Config box (to the right)
    draw_box(11, 8.4, 2.5, 1.2, 'Config File\n───────\ntransformMatrix\nLidarBoundary\ntransect (opt)',
             '#F3E5F5', fontsize=8)

    # =========== TRANSFORM SECTION ===========
    draw_box(6.5, 6.8, 3, 0.8, '1. Coordinate Transform\nScanner → UTM (NAD83)', colors['process'], fontsize=9)
    draw_arrow(8, 8.0, 8, 7.65)
    draw_arrow(11, 8.4, 9.55, 7.2)  # Config to transform

    # =========== SPLIT INTO L1 AND L2 ===========
    # Split point
    ax.plot(8, 6.4, 'ko', markersize=8)
    draw_arrow(8, 6.8, 8, 6.45)

    # L1 arrow (left)
    draw_arrow(8, 6.4, 4.5, 5.8)
    # L2 arrow (right)
    draw_arrow(8, 6.4, 11.5, 5.8)

    # =========== L1 PIPELINE (LEFT) ===========
    # L1 Header
    ax.text(3.5, 6.1, 'L1: Beach Surfaces', fontsize=12, fontweight='bold',
            color=colors['l1'], ha='center')

    # L1 Step 2
    draw_box(1.5, 4.8, 4, 0.8, '2. Spatial Filtering\nKeep points in boundary polygon',
             colors['process'], fontsize=9)
    draw_arrow(3.5, 5.75, 3.5, 5.65)

    # L1 Step 3
    draw_box(1.5, 3.6, 4, 0.9, '3. Spatial Binning (10cm grid)\nz_mean, z_mode, z_min, z_max,\nz_std, count per bin',
             colors['process'], fontsize=9)
    draw_arrow(3.5, 4.8, 3.5, 4.55)

    # L1 Output
    draw_box(1.5, 2.2, 4, 0.9, 'L1_YYYYMMDD.nc\n───────\n2D Grids: z(x, y)\nDaily beach DEMs',
             colors['output'], fontsize=9, bold=False)
    draw_arrow(3.5, 3.6, 3.5, 3.15)

    # L1 Use cases
    ax.text(3.5, 1.5, 'Use Cases:', fontsize=9, fontweight='bold', ha='center', color=colors['l1'])
    ax.text(3.5, 1.1, '• Beach morphology\n• Volume change\n• Foreshore slope (MSL→MHW)',
            fontsize=8, ha='center', va='top', color='#424242')

    # =========== L2 PIPELINE (RIGHT) ===========
    # L2 Header
    ax.text(12.5, 6.1, 'L2: Wave Dynamics', fontsize=12, fontweight='bold',
            color=colors['l2'], ha='center')

    # L2 Step 2
    draw_box(10.5, 4.8, 4, 0.8, '2. Transect Definition\nAuto-compute, manual, or MOP',
             colors['process'], fontsize=9)
    draw_arrow(12.5, 5.75, 12.5, 5.65)

    # L2 Step 3
    draw_box(10.5, 3.6, 4, 0.9, '3. Temporal Binning (2 Hz)\nX: 10cm along transect\nT: 0.5s time bins',
             colors['process'], fontsize=9)
    draw_arrow(12.5, 4.8, 12.5, 4.55)

    # L2 Output
    draw_box(10.5, 2.2, 4, 0.9, 'L2_YYYYMMDD.nc\n───────\nTimestacks: Z(x, t), I(x, t)\nWave-resolving',
             colors['output'], fontsize=9, bold=False)
    draw_arrow(12.5, 3.6, 12.5, 3.15)

    # L2 Use cases
    ax.text(12.5, 1.5, 'Use Cases:', fontsize=9, fontweight='bold', ha='center', color=colors['l2'])
    ax.text(12.5, 1.1, '• Wave runup detection\n• Swash dynamics\n• Runup statistics (R2%, Rmax)',
            fontsize=8, ha='center', va='top', color='#424242')

    # =========== LEGEND ===========
    legend_y = 0.3
    legend_patches = [
        mpatches.Patch(facecolor=colors['input'], edgecolor='#424242', label='Input Data'),
        mpatches.Patch(facecolor=colors['process'], edgecolor='#424242', label='Processing Step'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='#424242', label='Output'),
    ]
    ax.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=9,
              frameon=True, fancybox=True, shadow=False)

    # =========== SMALL DIAGRAMS ===========
    # L1 mini DEM
    ax_dem = fig.add_axes([0.08, 0.08, 0.12, 0.10])
    dem_data = np.random.rand(10, 10) * 2 + np.linspace(0, 2, 10).reshape(1, -1)
    dem_data = np.where(np.random.rand(10, 10) > 0.2, dem_data, np.nan)
    ax_dem.imshow(dem_data, cmap='terrain', origin='lower', aspect='equal')
    ax_dem.set_title('L1: DEM', fontsize=8, color=colors['l1'])
    ax_dem.set_xticks([])
    ax_dem.set_yticks([])
    ax_dem.set_xlabel('X', fontsize=7)
    ax_dem.set_ylabel('Y', fontsize=7)

    # L2 mini timestack
    ax_ts = fig.add_axes([0.80, 0.08, 0.12, 0.10])
    t = np.linspace(0, 10, 100)
    x = np.linspace(0, 50, 50)
    T, X = np.meshgrid(t, x)
    # Simulate wave runup pattern
    ts_data = 0.5 + 0.3 * np.sin(2 * np.pi * T / 5 - X / 10) + 0.1 * np.random.rand(*T.shape)
    ts_data = np.where(X < 20 + 10 * np.sin(2 * np.pi * T / 5), ts_data + 1, ts_data)
    ax_ts.imshow(ts_data, cmap='viridis', aspect='auto', origin='lower',
                 extent=[0, 10, 0, 50])
    ax_ts.set_title('L2: Timestack', fontsize=8, color=colors['l2'])
    ax_ts.set_xlabel('Time (s)', fontsize=7)
    ax_ts.set_ylabel('X (m)', fontsize=7)
    ax_ts.tick_params(labelsize=6)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Pipeline diagram saved to: {output_path}")


def create_slope_diagram(output_path: Path, dpi: int = 150):
    """Create a diagram showing the MSL-MHW slope calculation."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create beach profile
    x = np.linspace(0, 60, 200)
    # Typical beach profile: steep near water, flattening landward
    z = 0.3 + 0.05 * x - 0.0003 * x**2 + 0.1 * np.sin(x / 5)

    # Plot profile
    ax.fill_between(x, -0.5, z, color='#D2B48C', alpha=0.7, label='Beach')
    ax.plot(x, z, 'k-', linewidth=2)

    # Tidal datums
    z_msl = 0.744
    z_mhw = 1.34

    ax.axhline(y=z_msl, color='green', linestyle='--', linewidth=2, label=f'MSL ({z_msl}m)')
    ax.axhline(y=z_mhw, color='orange', linestyle='--', linewidth=2, label=f'MHW ({z_mhw}m)')

    # Find intersection points
    x_msl = x[np.argmin(np.abs(z - z_msl))]
    x_mhw = x[np.argmin(np.abs(z - z_mhw))]

    # Draw slope line
    ax.plot([x_msl, x_mhw], [z_msl, z_mhw], 'r-', linewidth=3, label='Foreshore Slope')
    ax.plot([x_msl, x_mhw], [z_msl, z_mhw], 'ro', markersize=8)

    # Annotations
    slope = (z_mhw - z_msl) / (x_mhw - x_msl)
    angle = np.degrees(np.arctan(slope))

    ax.annotate(f'Slope = {slope:.3f}\n({angle:.1f}°)',
                xy=((x_msl + x_mhw)/2, (z_msl + z_mhw)/2),
                xytext=(35, 2.5),
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

    # Water
    ax.fill_between([0, x_msl], -0.5, z_msl, color='#4FC3F7', alpha=0.5, label='Water')

    # Labels
    ax.set_xlabel('Cross-shore Distance (m)', fontsize=12)
    ax.set_ylabel('Elevation (m NAVD88)', fontsize=12)
    ax.set_title('Foreshore Slope Calculation: MSL to MHW', fontsize=14, fontweight='bold')

    ax.set_xlim(0, 60)
    ax.set_ylim(-0.5, 4)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add formula
    ax.text(0.02, 0.98,
            f'Slope = (MHW - MSL) / Δx\n     = ({z_mhw} - {z_msl}) / {x_mhw - x_msl:.1f}\n     = {slope:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Slope diagram saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline diagrams")
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output path for pipeline diagram (default: docs/pipeline_diagram.png)')
    parser.add_argument('--slope-output', type=Path, default=None,
                        help='Output path for slope diagram (default: docs/slope_diagram.png)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Resolution in DPI (default: 150)')

    args = parser.parse_args()

    # Determine output directory
    script_dir = Path(__file__).parent.parent.parent
    docs_dir = script_dir / 'docs'
    docs_dir.mkdir(exist_ok=True)

    # Pipeline diagram
    if args.output:
        pipeline_path = args.output
    else:
        pipeline_path = docs_dir / 'pipeline_diagram.png'

    create_pipeline_diagram(pipeline_path, dpi=args.dpi)

    # Slope diagram
    if args.slope_output:
        slope_path = args.slope_output
    else:
        slope_path = docs_dir / 'slope_diagram.png'

    create_slope_diagram(slope_path, dpi=args.dpi)

    print(f"\nDiagrams created in: {docs_dir}")


if __name__ == '__main__':
    main()
