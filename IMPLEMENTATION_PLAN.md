# Implementation Plan: Time-Crossshore Wave Runup Visualization (Z(x,t) Hovmöller Diagrams)

**Generated:** 2026-01-17
**Request:** Create time-vs-crossshore distance figures with intensity/softmax score colormaps and runup detection overlays
**Estimated Milestones:** 4
**Risk Level:** Medium - Requires implementing L2-style temporal processing not yet present in Python codebase

## Prerequisites
- [x] Python environment with numpy, scipy, xarray, pandas, laspy
- [ ] matplotlib added to requirements.txt for visualization
- [ ] Sample .laz data available for testing

## Architecture Decision Record
> **Approach:** Extend the existing 4-phase Python architecture by (1) adding L2-style temporal binning to phase2, (2) creating transect extraction functions in phase3, (3) adding visualization functions in a new phase5 module, and (4) creating an L2 orchestration function in phase4.
>
> **Key Trade-offs:**
> - **Chosen:** Build on existing phase architecture rather than direct MATLAB port → ensures consistency with established Python patterns and leverages existing config/filtering infrastructure
> - **Rejected:** Monolithic script approach → harder to test and maintain
> - **Chosen:** Use matplotlib for initial implementation → native Python, easy to test, widely supported
> - **Rejected:** plotly/bokeh → adds dependencies, but could be added later for interactivity
>
> **Failure Domains:**
> 1. **Temporal binning performance**: L2 processing with ~2Hz time resolution may be slow for large files → Mitigate by using vectorized numpy operations and optional downsampling parameter
> 2. **Memory usage**: Z(x,t) matrices can be large (e.g., 500 crossshore × 240 time steps = 120k floats) → Mitigate by processing files individually and using sparse representations where appropriate
> 3. **Runup detection accuracy**: Python implementation may differ from MATLAB due to moving minimum filter edge effects → Mitigate by comprehensive unit tests comparing against known outputs

---

## Milestone 1: Add Temporal Binning Infrastructure (L2-style Processing)

**Objective:** Extend phase2 to support time-resolved point cloud binning, producing Z(x,t) arrays along transects

**Why This Order:** Core data structures must exist before visualization or analysis can be implemented

### Changes
- [ ] `requirements.txt` - Add `matplotlib>=3.7` for visualization support
- [ ] `python/code/phase2.py` - Add `bin_point_cloud_temporal()` function that bins points in (x, y, t) space, returning time-indexed statistics similar to `accumpts_L2.m`
- [ ] `python/code/phase2.py` - Add `TimeResolvedGrid` dataclass to hold temporal binning results (x_edges, t_edges, z_vals, intensity_vals)
- [ ] `python/tests/test_phase2.py` - Add `test_bin_point_cloud_temporal()` that verifies temporal binning produces correct time slices and handles edge cases (empty bins, single time step)

### Verification Gate
```bash
pytest python/tests/test_phase2.py::test_bin_point_cloud_temporal -v
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] Test output contains: `PASSED`
- [ ] New function accepts (points, intensities, times) arrays and returns TimeResolvedGrid
- [ ] Time dimension correctly partitions points into ~2Hz bins (configurable frequency)

**Rollback if Failed:**
```bash
git checkout python/code/phase2.py python/tests/test_phase2.py requirements.txt
```

---

## Milestone 2: Implement Transect Extraction and Z(x,t) Matrix Generation

**Objective:** Create functions to extract cross-shore transects from 3D point clouds and build time-resolved elevation matrices

**Why This Order:** Transect extraction depends on temporal binning (Milestone 1) and is required before visualization (Milestone 3)

### Changes
- [ ] `python/code/phase3.py` - Add `extract_transect()` function that projects 3D points onto a shore-normal line, similar to `Get3_1Dprofiles.m` logic (uses line geometry to find points within tolerance, projects to 1D crossshore distance)
- [ ] `python/code/phase3.py` - Add `build_xt_matrix()` function that constructs Z(x,t) and I(x,t) matrices from temporal binning results along a transect
- [ ] `python/code/phase3.py` - Add `TransectConfig` dataclass to store transect geometry (start_utm, end_utm, alongshore_offset, tolerance)
- [ ] `python/tests/test_phase3.py` - Add `test_extract_transect()` verifying correct projection of 3D points to 1D crossshore positions
- [ ] `python/tests/test_phase3.py` - Add `test_build_xt_matrix()` checking Z(x,t) matrix has correct dimensions (n_crossshore × n_time) and handles NaN gaps

### Verification Gate
```bash
pytest python/tests/test_phase3.py::test_extract_transect python/tests/test_phase3.py::test_build_xt_matrix -v
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] Both tests pass
- [ ] `extract_transect()` correctly filters points within distance tolerance and computes crossshore positions
- [ ] `build_xt_matrix()` produces matrices matching expected shape with proper NaN handling for missing data

**Rollback if Failed:**
```bash
git checkout python/code/phase3.py python/tests/test_phase3.py
```

---

## Milestone 3: Create Visualization Functions (Hovmöller Diagrams)

**Objective:** Implement matplotlib-based plotting functions to create time-crossshore figures with intensity/elevation colormaps

**Why This Order:** Requires Z(x,t) matrices from Milestone 2; visualization is the primary user-facing deliverable

### Changes
- [ ] `python/code/phase5.py` - Create new module for L2 analysis and visualization
- [ ] `python/code/phase5.py` - Add `plot_hovmoller()` function that creates pcolormesh plots of Z(x,t) with configurable colormap, labels, and color limits matching the reference image format
- [ ] `python/code/phase5.py` - Add `overlay_runup_line()` function to plot runup detection results (solid line for optimized, dotted for bounds) over Hovmöller diagram
- [ ] `python/code/phase5.py` - Add `create_dual_panel_figure()` helper to create stacked subplots: (a) intensity and (b) softmax score/elevation, matching the reference figure layout
- [ ] `python/tests/test_phase5.py` - Add `test_plot_hovmoller()` that verifies figure creation without errors, checks axes labels, and validates colorbar presence
- [ ] `python/tests/test_phase5.py` - Add `test_overlay_runup_line()` checking line plotting with different styles (solid, dotted) and data ranges

### Verification Gate
```bash
pytest python/tests/test_phase5.py -v
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] All tests pass
- [ ] Generated figures have x-axis (time in seconds), y-axis (crossshore distance in meters)
- [ ] Colorbar displays with appropriate label
- [ ] No matplotlib warnings about incompatible dimensions

**Rollback if Failed:**
```bash
git rm python/code/phase5.py python/tests/test_phase5.py
```

---

## Milestone 4: Add L2 Orchestration and CLI Integration

**Objective:** Create high-level `process_l2()` function and integrate with existing pipeline structure for end-to-end L2 processing

**Why This Order:** Final integration milestone that composes all previous components into user-accessible workflow

### Changes
- [ ] `python/code/phase4.py` - Add `process_l2()` function that orchestrates: file discovery → temporal binning → transect extraction → Z(x,t) matrix building, returning both matrices and metadata
- [ ] `python/code/phase4.py` - Add `save_l2_results()` function to persist Z(x,t) matrices and metadata to NetCDF with proper coordinate labels (time, crossshore_distance)
- [ ] `python/code/phase5.py` - Add `detect_runup_line()` function implementing moving minimum filter and threshold intersection logic from `get_runupStats_L2.m` lines 42-100
- [ ] `python/tests/test_phase4.py` - Add `test_process_l2()` with synthetic data simulating wave runup (known sinusoidal Z(x,t) pattern)
- [ ] `python/tests/test_phase5.py` - Add `test_detect_runup_line()` verifying runup detection finds correct intersection points for synthetic water level curves
- [ ] `README.md` - Add "L2 Processing (Wave Runup)" section documenting `process_l2()` usage with example code snippet

### Verification Gate
```bash
pytest python/tests/test_phase4.py::test_process_l2 python/tests/test_phase5.py::test_detect_runup_line -v
```

**Success Criteria:**
- [ ] Exit code: `0`
- [ ] Both tests pass
- [ ] `process_l2()` successfully chains all L2 processing steps
- [ ] Synthetic runup detection test recovers known runup positions within 0.1m tolerance
- [ ] NetCDF output contains expected variables: elevation, intensity, time, crossshore_distance

**Rollback if Failed:**
```bash
git checkout python/code/phase4.py python/code/phase5.py python/tests/test_phase4.py python/tests/test_phase5.py README.md
```

---

## Final Verification

```bash
# Full test suite
pytest python/tests/ -v --tb=short

# Generate example figure from synthetic data
python -c "
import numpy as np
import sys
sys.path.append('python/code')
import phase5

# Create synthetic Z(x,t) data
x = np.linspace(-100, 0, 200)
t = np.linspace(0, 120, 240)
X, T = np.meshgrid(x, t)
Z = 1.5 + 0.3 * np.sin(2*np.pi*T/20) * np.exp(X/30)  # Simulated waves

fig, ax = phase5.plot_hovmoller(t, x, Z.T, cmap='viridis',
                                 xlabel='time (s)', ylabel='x-shore distance (m)',
                                 clabel='elevation (m)')
fig.savefig('example_hovmoller.png', dpi=150, bbox_inches='tight')
print('Figure saved to example_hovmoller.png')
"
```

**Definition of Done:**
- [ ] All unit tests pass (phases 2, 3, 4, 5)
- [ ] Example Hovmöller diagram generated successfully matches reference image structure (time on x-axis, crossshore on y-axis, colormap for elevation/intensity)
- [ ] No regressions: `pytest python/tests/test_phase1.py python/tests/test_phase2.py python/tests/test_phase3.py -v` all pass
- [ ] Documentation includes L2 usage example
- [ ] Generated NetCDF files can be opened and inspected with xarray

## Appendix: File Change Summary
| File | Action | Milestone |
|------|--------|-----------|
| `requirements.txt` | Modify | 1 |
| `python/code/phase2.py` | Modify | 1 |
| `python/tests/test_phase2.py` | Modify | 1 |
| `python/code/phase3.py` | Modify | 2 |
| `python/tests/test_phase3.py` | Modify | 2 |
| `python/code/phase5.py` | Create | 3, 4 |
| `python/tests/test_phase5.py` | Create | 3, 4 |
| `python/code/phase4.py` | Modify | 4 |
| `python/tests/test_phase4.py` | Modify | 4 |
| `README.md` | Modify | 4 |

## Implementation Notes

### Temporal Binning Strategy (Milestone 1)
The `bin_point_cloud_temporal()` function will:
1. Quantize GPS times to desired frequency (default ~2Hz via `ff` parameter, matching L2_pipeline.m line 133)
2. Bin points into (x, y, t) voxels using 3D accumarray-style logic
3. Compute statistics per voxel: mean, min, max for Z and I
4. Return TimeResolvedGrid with edges and value arrays

### Transect Geometry (Milestone 2)
Following `Get3_1Dprofiles.m` approach:
- Define shore-normal transect via start/end UTM coordinates
- Compute perpendicular distance from each point to transect line
- Keep points within tolerance (default 1m)
- Project points onto transect to get crossshore position
- Support multiple parallel transects via alongshore_offset parameter

### Runup Detection Algorithm (Milestone 4)
Python implementation of `get_runupStats_L2.m` logic:
1. Apply moving minimum filter to Z(x,t) along time axis (window = IGlength/dt samples)
2. Compute water level: `wlev = Z(x,t) - moving_min(Z(x,t))`
3. For each time step, find crossshore position where `wlev` crosses threshold (default 0.1m)
4. Apply median filtering and gap interpolation to smooth detection
5. Return arrays: Xrunup(t), Zrunup(t)

### Figure Aesthetics
To match reference image:
- Use `pcolormesh` for efficient rendering of Z(x,t) matrix
- Apply `shading='auto'` or `'gouraud'` for smooth appearance
- Set colormap: `'gray'` for intensity (panel a), `'RdBu_r'` for softmax/elevation (panel b)
- Configure color limits via `clim` parameter (vmin, vmax)
- Use solid line (`'-'`) for optimized runup, dotted (`':'`) for bounds
- Add grid lines: `ax.grid(True, alpha=0.3)`
