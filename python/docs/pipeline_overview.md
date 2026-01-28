# Laserdanger Pipeline Overview

This document explains the L1 and L2 processing pipelines for the Livox Avia LiDAR data.

---

## High-Level Overview

```
                         RAW DATA
                            |
                    .laz files (timestamped)
                    from Livox Avia scanner
                            |
            +---------------+---------------+
            |                               |
            v                               v
    +---------------+               +---------------+
    |  L1 Pipeline  |               |  L2 Pipeline  |
    |  (Surfaces)   |               |  (Timestacks) |
    +---------------+               +---------------+
            |                               |
            v                               v
    Daily Beach DEMs              Wave Runup Dynamics
    (morphology)                  (swash zone)
```

---

## What the Scanner Sees

The Livox Avia scanner sits on a tower overlooking the beach and continuously scans the surface:

```
    Scanner (on tower)
         /\
        /  \
       /    \  <- LiDAR beam sweeps back and forth
      /      \
     /        \
    /          \
   +-----------+--~-~-~-~-~-~
   |  BEACH    |    OCEAN
   |  (dry)    |  (wet/swash)
   +-----------+--~-~-~-~-~-~

   |<-- 50-80m coverage -->|
```

Each .laz file contains ~30 seconds of point cloud data with:
- X, Y, Z coordinates (in scanner reference frame)
- Intensity (reflectance)
- Timestamp

---

## L1 Pipeline: Beach Surface Morphology

**Purpose:** Create daily gridded beach surface maps (DEMs)

**Use Cases:**
- Beach erosion/accretion monitoring
- Volume change calculations
- Dry beach slope analysis

### L1 Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         L1 PROCESSING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

   .laz files (one day)
   ├── file_20260120_200000.laz
   ├── file_20260120_200030.laz
   ├── file_20260120_200100.laz
   └── ... (hundreds of files)
            │
            ▼
┌─────────────────────────────────────┐
│  1. COORDINATE TRANSFORM            │
│  ─────────────────────────────────  │
│  Scanner coords → UTM (NAD83)       │
│  Uses 4x4 transformation matrix     │
│  from config file                   │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  2. SPATIAL FILTERING               │
│  ─────────────────────────────────  │
│  Keep only points inside            │
│  LidarBoundary polygon              │
│  (removes buildings, vegetation)    │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  3. SPATIAL BINNING                 │
│  ─────────────────────────────────  │
│  Grid points into 10cm x 10cm bins  │
│  Calculate per-bin statistics:      │
│  • z_mean  (average elevation)      │
│  • z_mode  (most common elevation)  │
│  • z_min, z_max (range)             │
│  • z_std   (variability)            │
│  • count   (point density)          │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  4. OUTPUT                          │
│  ─────────────────────────────────  │
│  L1_20260120.nc (NetCDF)            │
│  • 2D grids: z(x, y)                │
│  • Multiple time slices per day     │
└─────────────────────────────────────┘
```

### L1 Output Visualization

```
        L1 DEM (Bird's Eye View)
    ┌─────────────────────────────┐
    │  ████████████              │  ← High elevation (dry beach)
    │  ████████████████          │
    │  ████████████████████      │
    │  ██████████████████████    │
    │  ████████████████████████  │  ← Transition zone
    │  ██████████████████████    │
    │  ████████████████          │
    │  ██████████                │  ← Low elevation (wet sand)
    │  ████                      │
    │                            │  ← Water (no returns)
    └─────────────────────────────┘
      Y ↑
        → X (cross-shore)
```

---

## L2 Pipeline: Wave-Resolving Timestacks

**Purpose:** Capture high-frequency (2 Hz) water surface dynamics along a cross-shore transect

**Use Cases:**
- Wave runup detection
- Swash zone dynamics
- Runup statistics (R2%, Rmax, etc.)

### L2 Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         L2 PROCESSING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

   .laz files (one day)
   ├── file_20260120_200000.laz
   ├── file_20260120_200030.laz
   └── ...
            │
            ▼
┌─────────────────────────────────────┐
│  1. COORDINATE TRANSFORM            │
│  ─────────────────────────────────  │
│  Scanner coords → UTM               │
│  (same as L1)                       │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  2. TRANSECT DEFINITION             │
│  ─────────────────────────────────  │
│  Define cross-shore transect line   │
│  Options:                           │
│  • Auto-compute from data           │
│  • Manual (config file)             │
│  • MOP transect (--auto-mop)        │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  3. TRANSECT EXTRACTION             │
│  ─────────────────────────────────  │
│  Filter points within tolerance     │
│  of transect line (±1-2m)           │
│  Project onto 1D cross-shore axis   │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  4. TEMPORAL BINNING                │
│  ─────────────────────────────────  │
│  Bin into space-time grid:          │
│  • X bins: 10cm along transect      │
│  • T bins: 0.5s (2 Hz)              │
│  Calculate Z(x,t) and I(x,t)        │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  5. OUTPUT                          │
│  ─────────────────────────────────  │
│  L2_20260120.nc (NetCDF)            │
│  • Z(x, t): Elevation timestack     │
│  • I(x, t): Intensity timestack     │
└─────────────────────────────────────┘
```

### L2 Timestack Visualization

```
    Elevation Timestack Z(x, t)

    Distance (x) →
    0m        20m        40m        60m
    │          │          │          │
    ├──────────┴──────────┴──────────┤ ← Seaward
    │▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ t=0s
    │░▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░│
    │░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░│   Wave running up
    │░░░▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░│      ↓
    │░░░░▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░│
    │░░░░░▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░│  ← Runup maximum
    │░░░░░░▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░│
    │░░░░░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░│   Wave receding
    │░░░░░░░░▓▓░░░░░░░░░░░░░░░░░░░░░│      ↓
    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
    │▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ ← Next wave arrives
    │░▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░│
    └───────────────────────────────┘ ← Landward

    Time (t) ↓

    Legend: ▓ = Water (high Z)  ░ = Dry beach (low Z)
```

---

## MOP Transects (California Coast)

MOP (Monitoring and Prediction) transects are standardized cross-shore survey lines spaced ~100m apart along the California coast.

```
    California Coastline with MOP Transects

         N
         ↑
    ─────┼─────────────────────────
         │    OCEAN
         │
    ═════╪═════  MOP 512
         │
    ═════╪═════  MOP 511
         │
    ═════╪═════  MOP 510  ← Scanner here (Tower site)
         │
    ═════╪═════  MOP 509
         │
    ─────┴─────────────────────────
         LAND

    Using --auto-mop automatically selects the best MOP
    for your scanner location (e.g., MOP 510 for Tower)
```

---

## Slope Calculation (L1)

The foreshore slope is calculated between tidal datum elevations:

```
    Cross-Shore Profile

    Elevation (m NAVD88)
         │
    2.0  │                          ████
         │                      ████
    1.5  │- - - - - - - - - ████- - - - -  ← MHW (1.34m)
         │              ████ ↑
    1.0  │          ████     │ SLOPE
         │- - - ████ - - - - ↓ - - - - -  ← MSL (0.744m)
    0.5  │  ████
         │██
    0.0  └────────────────────────────────→ Distance (m)
         0    10    20    30    40    50

         Seaward ←──────────────→ Landward

    Slope = rise / run = (MHW - MSL) / horizontal distance
          = (1.34 - 0.744) / Δx
```

---

## Quick Reference: CLI Commands

### L1 Processing
```bash
# Process all days
python scripts/processing/run_daily_l1.py --config configs/towr_livox_config_20260120.json

# Process specific dates
python scripts/processing/run_daily_l1.py --config configs/towr_livox_config_20260120.json \
    --start 2026-01-20 --end 2026-01-21

# Preview (dry run)
python scripts/processing/run_daily_l1.py --config configs/towr_livox_config_20260120.json --dry-run
```

### L2 Processing
```bash
# Process all days with auto-MOP selection
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json --auto-mop

# Process with specific MOP
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json --mop 510

# Memory-efficient chunked processing
python scripts/processing/run_daily_l2.py --config configs/towr_livox_config_20260120.json \
    --chunk-size 10 --auto-mop
```

### Visualization
```bash
# L1 GIF with slope (uses MSL→MHW by default)
python scripts/visualization/gif_nc_l1.py --config configs/towr_livox_config_20260120.json

# L2 timestack visualization
python scripts/visualization/visualize_l2.py --config configs/towr_livox_config_20260120.json
```

---

## File Naming Convention

```
Input:   {sensor}_{YYYYMMDD}_{HHMMSS}.laz
         └─ Raw scanner data with timestamp

L1:      L1_{YYYYMMDD}.nc
         L1_{YYYYMMDD}_MOP510.nc        (with MOP transect)
         └─ Daily gridded surface

L2:      L2_{YYYYMMDD}.nc
         L2_{YYYYMMDD}_MOP510.nc        (with MOP transect)
         L2_{YYYYMMDD}_exp02.nc         (with expansion rate)
         └─ Daily timestack
```

---

## Data Flow Summary

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│   Scanner          Config              Processing           Output         │
│   ────────         ──────              ──────────           ──────         │
│                                                                            │
│   Livox     ──→   transform    ──→    L1 Pipeline    ──→   DEMs           │
│   Avia           matrix                (surfaces)          z(x,y)         │
│     │              │                                          │            │
│     │              │                                          ▼            │
│     │              │                                    Beach Morphology   │
│     │              │                                    Slope Analysis     │
│     │              │                                                       │
│     │              │                                                       │
│     │              ▼                                                       │
│     └────────→   transect      ──→    L2 Pipeline    ──→   Timestacks     │
│                  config               (dynamics)           Z(x,t)         │
│                  (or MOP)                                     │            │
│                                                               ▼            │
│                                                         Wave Runup        │
│                                                         Swash Stats       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Glossary

| Term | Definition |
|------|------------|
| **LAZ** | Compressed LAS format for point cloud data |
| **DEM** | Digital Elevation Model - gridded surface |
| **Timestack** | Space-time matrix Z(x,t) showing surface evolution |
| **MOP** | Monitoring and Prediction - standardized CA coast transects |
| **MSL** | Mean Sea Level (0.744m NAVD88 at this site) |
| **MHW** | Mean High Water (1.34m NAVD88 at this site) |
| **NAVD88** | North American Vertical Datum of 1988 |
| **UTM** | Universal Transverse Mercator coordinate system |
| **Swash** | The wave uprush/backwash zone on the beach |
| **Runup** | Maximum vertical extent of wave uprush |

---

## Need Help?

- Check `python/CLAUDE.md` for detailed CLI reference
- Run any script with `--help` for options
- Use `--dry-run` to preview without processing
