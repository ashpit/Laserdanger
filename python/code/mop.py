"""
MOP (Monitoring and Prediction) transect integration for L2 processing.

Provides functionality to:
- Load MOP table from CSV
- Get integer or fractional MOP transects
- Select optimal MOP for a given point cloud
- Convert MOP transects to TransectConfig for processing
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import profiles

logger = logging.getLogger(__name__)

# Default path to MOP table relative to the code directory
DEFAULT_MOP_TABLE_PATH = Path(__file__).parent.parent / "mop_data" / "MopTable.csv"


@dataclass(frozen=True)
class MopTransect:
    """
    Single MOP transect definition.

    Attributes
    ----------
    mop_num : Union[int, float]
        MOP number (can be fractional for interpolated transects)
    back_x : float
        Backshore X coordinate (UTM)
    back_y : float
        Backshore Y coordinate (UTM)
    off_x : float
        Offshore X coordinate (UTM)
    off_y : float
        Offshore Y coordinate (UTM)
    """
    mop_num: Union[int, float]
    back_x: float
    back_y: float
    off_x: float
    off_y: float

    @property
    def length(self) -> float:
        """Calculate transect length in meters."""
        dx = self.off_x - self.back_x
        dy = self.off_y - self.back_y
        return math.sqrt(dx * dx + dy * dy)

    @property
    def azimuth(self) -> float:
        """
        Calculate transect azimuth (degrees from north, clockwise).

        Returns direction from backshore to offshore.
        """
        dx = self.off_x - self.back_x
        dy = self.off_y - self.back_y
        # atan2 gives angle from east, counterclockwise
        # Convert to azimuth (from north, clockwise)
        angle_rad = math.atan2(dx, dy)  # Note: atan2(dx, dy) for azimuth
        azimuth_deg = math.degrees(angle_rad)
        if azimuth_deg < 0:
            azimuth_deg += 360
        return azimuth_deg

    def to_transect_config(
        self,
        tolerance: float = 2.0,
        resolution: float = 0.1,
        expansion_rate: float = 0.0,
        alongshore_spacings: Tuple[float, ...] = (0,),
    ) -> profiles.TransectConfig:
        """
        Convert MOP transect to TransectConfig for L2 processing.

        Parameters
        ----------
        tolerance : float
            Base transect tolerance in meters (default 2.0)
        resolution : float
            Cross-shore resolution in meters (default 0.1)
        expansion_rate : float
            Adaptive tolerance expansion rate (default 0.0)
        alongshore_spacings : tuple
            Alongshore offsets for multiple transects (default single central)

        Returns
        -------
        TransectConfig
            Configuration ready for profile extraction
        """
        return profiles.TransectConfig(
            x1=self.back_x,
            y1=self.back_y,
            x2=self.off_x,
            y2=self.off_y,
            tolerance=tolerance,
            resolution=resolution,
            expansion_rate=expansion_rate,
            alongshore_spacings=alongshore_spacings,
            extend_line=(0, 0),  # No extension needed, MOP endpoints are fixed
        )


@dataclass
class MopTable:
    """
    MOP table manager for loading and querying MOP transects.

    The MOP table contains transect endpoints for California's
    Monitoring and Prediction (MOP) system.
    """
    data: pd.DataFrame
    _path: Optional[Path] = None

    def __post_init__(self):
        """Validate the loaded data."""
        required_cols = ['MopNum', 'BackXutm', 'BackYutm', 'OffXutm', 'OffYutm']
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"MOP table missing required columns: {missing}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "MopTable":
        """
        Load MOP table from CSV file.

        Parameters
        ----------
        path : Path, optional
            Path to MOP CSV file. If None, uses default location.

        Returns
        -------
        MopTable
            Loaded MOP table manager

        Raises
        ------
        FileNotFoundError
            If the MOP table file doesn't exist
        ValueError
            If the file is missing required columns
        """
        if path is None:
            path = DEFAULT_MOP_TABLE_PATH

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MOP table not found: {path}")

        logger.debug("Loading MOP table from %s", path)
        df = pd.read_csv(path)

        # Filter out rows with NaN MOP numbers
        initial_count = len(df)
        df = df.dropna(subset=['MopNum'])
        df['MopNum'] = df['MopNum'].astype(int)
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.debug("Filtered %d rows with missing MOP numbers", filtered_count)

        return cls(data=df, _path=path)

    @property
    def mop_numbers(self) -> List[int]:
        """Get list of all MOP numbers in the table."""
        return sorted(self.data['MopNum'].tolist())

    @property
    def min_mop(self) -> int:
        """Minimum MOP number in table."""
        return int(self.data['MopNum'].min())

    @property
    def max_mop(self) -> int:
        """Maximum MOP number in table."""
        return int(self.data['MopNum'].max())

    def __len__(self) -> int:
        """Number of MOPs in table."""
        return len(self.data)

    def get_mop(self, mop_num: int) -> MopTransect:
        """
        Get transect for a specific integer MOP number.

        Parameters
        ----------
        mop_num : int
            MOP number to retrieve

        Returns
        -------
        MopTransect
            Transect definition for this MOP

        Raises
        ------
        KeyError
            If MOP number not found in table
        """
        row = self.data[self.data['MopNum'] == mop_num]
        if len(row) == 0:
            raise KeyError(f"MOP {mop_num} not found in table (range: {self.min_mop}-{self.max_mop})")

        row = row.iloc[0]
        return MopTransect(
            mop_num=int(mop_num),
            back_x=float(row['BackXutm']),
            back_y=float(row['BackYutm']),
            off_x=float(row['OffXutm']),
            off_y=float(row['OffYutm']),
        )

    def get_fractional_mop(self, mop_num: float) -> MopTransect:
        """
        Get transect for a fractional MOP number via linear interpolation.

        Interpolates between adjacent MOPs to create a transect at the
        specified fractional position.

        Parameters
        ----------
        mop_num : float
            Fractional MOP number (e.g., 456.3 interpolates between 456 and 457)

        Returns
        -------
        MopTransect
            Interpolated transect definition

        Raises
        ------
        KeyError
            If base or next MOP not found in table
        ValueError
            If mop_num is outside valid range

        Examples
        --------
        >>> table = MopTable.load()
        >>> mop = table.get_fractional_mop(456.3)
        >>> # Returns transect 30% of the way between MOP 456 and 457
        """
        # Check if it's actually an integer
        if mop_num == int(mop_num):
            return self.get_mop(int(mop_num))

        base_mop = int(math.floor(mop_num))
        frac = mop_num - base_mop
        next_mop = base_mop + 1

        # Validate range
        if base_mop < self.min_mop or next_mop > self.max_mop:
            raise ValueError(
                f"MOP {mop_num} out of valid range [{self.min_mop}, {self.max_mop}]"
            )

        # Get both MOPs
        mop1 = self.get_mop(base_mop)
        mop2 = self.get_mop(next_mop)

        # Linear interpolation
        back_x = mop1.back_x + frac * (mop2.back_x - mop1.back_x)
        back_y = mop1.back_y + frac * (mop2.back_y - mop1.back_y)
        off_x = mop1.off_x + frac * (mop2.off_x - mop1.off_x)
        off_y = mop1.off_y + frac * (mop2.off_y - mop1.off_y)

        logger.debug(
            "Interpolated MOP %.1f from MOPs %d and %d (frac=%.2f)",
            mop_num, base_mop, next_mop, frac
        )

        return MopTransect(
            mop_num=mop_num,
            back_x=back_x,
            back_y=back_y,
            off_x=off_x,
            off_y=off_y,
        )

    def find_mops_in_bounds(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> List[int]:
        """
        Find MOPs whose endpoints fall within given UTM bounds.

        A MOP is included if either its backshore or offshore endpoint
        is within the specified bounding box.

        Parameters
        ----------
        x_min, x_max : float
            X (easting) bounds in UTM
        y_min, y_max : float
            Y (northing) bounds in UTM

        Returns
        -------
        List[int]
            MOP numbers within bounds, sorted ascending
        """
        df = self.data

        # Check if any endpoint is within bounds
        back_in_bounds = (
            (df['BackXutm'] >= x_min) & (df['BackXutm'] <= x_max) &
            (df['BackYutm'] >= y_min) & (df['BackYutm'] <= y_max)
        )
        off_in_bounds = (
            (df['OffXutm'] >= x_min) & (df['OffXutm'] <= x_max) &
            (df['OffYutm'] >= y_min) & (df['OffYutm'] <= y_max)
        )

        in_bounds = back_in_bounds | off_in_bounds
        mops = sorted(df.loc[in_bounds, 'MopNum'].astype(int).tolist())

        logger.debug(
            "Found %d MOPs in bounds (%.0f-%.0f, %.0f-%.0f)",
            len(mops), x_min, x_max, y_min, y_max
        )

        return mops


def _distance_point_to_line(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> Tuple[float, float]:
    """
    Calculate perpendicular distance from point to line segment.

    Also returns the signed cross-shore distance along the line.

    Parameters
    ----------
    px, py : float
        Point coordinates
    x1, y1 : float
        Line start point (backshore)
    x2, y2 : float
        Line end point (offshore)

    Returns
    -------
    perp_dist : float
        Perpendicular distance from point to line
    xshore_dist : float
        Distance along line from start to perpendicular projection
    """
    # Line vector
    dx = x2 - x1
    dy = y2 - y1
    line_length = math.sqrt(dx * dx + dy * dy)

    if line_length < 1e-10:
        return math.sqrt((px - x1)**2 + (py - y1)**2), 0.0

    # Unit vector along line
    ux = dx / line_length
    uy = dy / line_length

    # Vector from line start to point
    vx = px - x1
    vy = py - y1

    # Projection onto line
    xshore_dist = vx * ux + vy * uy

    # Perpendicular distance
    perp_dist = abs(vx * uy - vy * ux)

    return perp_dist, xshore_dist


def select_best_mop(
    X: np.ndarray,
    Y: np.ndarray,
    mop_table: MopTable,
    scanner_position: Optional[Tuple[float, float]] = None,
    method: str = "centroid",
    tolerance: float = 50.0,
) -> int:
    """
    Select the best MOP for a given point cloud.

    Parameters
    ----------
    X, Y : array
        Point cloud coordinates (UTM)
    mop_table : MopTable
        Loaded MOP table
    scanner_position : tuple (x, y), optional
        Scanner position in UTM (used for "nearest_scanner" method)
    method : str
        Selection method:
        - "centroid" (default): MOP closest to data centroid
        - "coverage": MOP with most points within tolerance
        - "nearest_scanner": MOP closest to scanner position
    tolerance : float
        Tolerance for coverage calculation (meters, default 50m)

    Returns
    -------
    int
        Selected MOP number

    Raises
    ------
    ValueError
        If no suitable MOP found
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()

    # Remove invalid values
    valid = ~(np.isnan(X) | np.isnan(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) == 0:
        raise ValueError("No valid points for MOP selection")

    # Get bounding box with some padding for MOP search
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    padding = 500  # meters

    # Find MOPs in the region
    candidate_mops = mop_table.find_mops_in_bounds(
        x_min - padding, x_max + padding,
        y_min - padding, y_max + padding,
    )

    if not candidate_mops:
        # Expand search
        logger.warning("No MOPs in immediate vicinity, searching wider area")
        candidate_mops = mop_table.find_mops_in_bounds(
            x_min - 2000, x_max + 2000,
            y_min - 2000, y_max + 2000,
        )

    if not candidate_mops:
        raise ValueError(
            f"No MOPs found near data (bounds: {x_min:.0f}-{x_max:.0f}, {y_min:.0f}-{y_max:.0f})"
        )

    logger.debug("Evaluating %d candidate MOPs: %s", len(candidate_mops), candidate_mops)

    if method == "centroid":
        return _select_by_centroid(X, Y, mop_table, candidate_mops)
    elif method == "coverage":
        return _select_by_coverage(X, Y, mop_table, candidate_mops, tolerance)
    elif method == "nearest_scanner":
        if scanner_position is None:
            raise ValueError("Scanner position required for 'nearest_scanner' method")
        return _select_by_scanner(scanner_position, mop_table, candidate_mops)
    else:
        raise ValueError(f"Unknown selection method: {method}")


def _select_by_centroid(
    X: np.ndarray,
    Y: np.ndarray,
    mop_table: MopTable,
    candidate_mops: List[int],
) -> int:
    """Select MOP closest to data centroid."""
    centroid_x = np.mean(X)
    centroid_y = np.mean(Y)

    best_mop = None
    best_dist = float('inf')

    for mop_num in candidate_mops:
        mop = mop_table.get_mop(mop_num)

        # Distance from centroid to MOP line
        dist, _ = _distance_point_to_line(
            centroid_x, centroid_y,
            mop.back_x, mop.back_y,
            mop.off_x, mop.off_y,
        )

        if dist < best_dist:
            best_dist = dist
            best_mop = mop_num

    logger.info(
        "Selected MOP %d (closest to centroid, dist=%.1fm)",
        best_mop, best_dist
    )

    return best_mop


def _select_by_coverage(
    X: np.ndarray,
    Y: np.ndarray,
    mop_table: MopTable,
    candidate_mops: List[int],
    tolerance: float,
) -> int:
    """Select MOP with most points within tolerance."""
    best_mop = None
    best_count = 0

    for mop_num in candidate_mops:
        mop = mop_table.get_mop(mop_num)

        # Count points within tolerance of MOP line
        count = 0
        for px, py in zip(X, Y):
            dist, _ = _distance_point_to_line(
                px, py,
                mop.back_x, mop.back_y,
                mop.off_x, mop.off_y,
            )
            if dist <= tolerance:
                count += 1

        if count > best_count:
            best_count = count
            best_mop = mop_num

    coverage_pct = 100 * best_count / len(X) if len(X) > 0 else 0
    logger.info(
        "Selected MOP %d (best coverage: %d points, %.1f%%)",
        best_mop, best_count, coverage_pct
    )

    return best_mop


def _select_by_scanner(
    scanner_position: Tuple[float, float],
    mop_table: MopTable,
    candidate_mops: List[int],
) -> int:
    """Select MOP closest to scanner position."""
    scanner_x, scanner_y = scanner_position
    best_mop = None
    best_dist = float('inf')

    for mop_num in candidate_mops:
        mop = mop_table.get_mop(mop_num)

        # Distance from scanner to MOP line
        dist, _ = _distance_point_to_line(
            scanner_x, scanner_y,
            mop.back_x, mop.back_y,
            mop.off_x, mop.off_y,
        )

        if dist < best_dist:
            best_dist = dist
            best_mop = mop_num

    logger.info(
        "Selected MOP %d (closest to scanner, dist=%.1fm)",
        best_mop, best_dist
    )

    return best_mop


def utm2mop_xshore(
    x: np.ndarray,
    y: np.ndarray,
    mop_table: MopTable,
    mop_num: Optional[Union[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert UTM coordinates to MOP cross-shore distances.

    Projects each point onto the nearest MOP transect and returns
    the cross-shore distance along that transect.

    Parameters
    ----------
    x, y : array
        UTM coordinates
    mop_table : MopTable
        Loaded MOP table
    mop_num : int or float, optional
        Specific MOP to use. If None, finds nearest MOP for each point.

    Returns
    -------
    mop_nums : array
        MOP number for each point (fractional if interpolated)
    xshore : array
        Cross-shore distance along the MOP transect (0 = backshore)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    n = len(x)
    mop_nums = np.full(n, np.nan)
    xshore = np.full(n, np.nan)

    if mop_num is not None:
        # Use specified MOP for all points
        if mop_num != int(mop_num):
            mop = mop_table.get_fractional_mop(mop_num)
        else:
            mop = mop_table.get_mop(int(mop_num))

        for i in range(n):
            _, xs = _distance_point_to_line(
                x[i], y[i],
                mop.back_x, mop.back_y,
                mop.off_x, mop.off_y,
            )
            mop_nums[i] = mop_num
            xshore[i] = xs

    else:
        # Find nearest MOP for each point
        all_mops = mop_table.mop_numbers

        for i in range(n):
            best_mop = None
            best_dist = float('inf')
            best_xshore = 0.0

            for mop_num_iter in all_mops:
                mop = mop_table.get_mop(mop_num_iter)
                dist, xs = _distance_point_to_line(
                    x[i], y[i],
                    mop.back_x, mop.back_y,
                    mop.off_x, mop.off_y,
                )

                if dist < best_dist:
                    best_dist = dist
                    best_mop = mop_num_iter
                    best_xshore = xs

            if best_mop is not None:
                mop_nums[i] = best_mop
                xshore[i] = best_xshore

    return mop_nums, xshore


def get_mop_transect(
    mop_num: Union[int, float],
    mop_table: Optional[MopTable] = None,
    mop_table_path: Optional[Path] = None,
    tolerance: float = 2.0,
    resolution: float = 0.1,
    expansion_rate: float = 0.0,
) -> profiles.TransectConfig:
    """
    Convenience function to get a TransectConfig from a MOP number.

    Parameters
    ----------
    mop_num : int or float
        MOP number (supports fractional for interpolation)
    mop_table : MopTable, optional
        Pre-loaded MOP table. If None, loads from mop_table_path.
    mop_table_path : Path, optional
        Path to MOP CSV file. Uses default if None.
    tolerance : float
        Base transect tolerance in meters (default 2.0)
    resolution : float
        Cross-shore resolution in meters (default 0.1)
    expansion_rate : float
        Adaptive tolerance expansion rate (default 0.0)

    Returns
    -------
    TransectConfig
        Configuration ready for L2 processing
    """
    if mop_table is None:
        mop_table = MopTable.load(mop_table_path)

    if mop_num != int(mop_num):
        mop = mop_table.get_fractional_mop(mop_num)
    else:
        mop = mop_table.get_mop(int(mop_num))

    return mop.to_transect_config(
        tolerance=tolerance,
        resolution=resolution,
        expansion_rate=expansion_rate,
    )


def format_mop_filename_suffix(mop_num: Union[int, float, None]) -> str:
    """
    Format MOP number for use in output filenames.

    Parameters
    ----------
    mop_num : int, float, or None
        MOP number (can be fractional)

    Returns
    -------
    str
        Formatted suffix (e.g., "_MOP456" or "_MOP456.3") or empty string

    Examples
    --------
    >>> format_mop_filename_suffix(456)
    '_MOP456'
    >>> format_mop_filename_suffix(456.3)
    '_MOP456.3'
    >>> format_mop_filename_suffix(None)
    ''
    """
    if mop_num is None:
        return ""

    if mop_num == int(mop_num):
        return f"_MOP{int(mop_num)}"
    else:
        return f"_MOP{mop_num:.1f}"
