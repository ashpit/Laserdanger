"""Tests for MOP (Monitoring and Prediction) transect module."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))
import mop  # noqa: E402
import profiles  # noqa: E402


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mop_table():
    """Load the real MOP table for testing."""
    return mop.MopTable.load()


@pytest.fixture
def sample_point_cloud():
    """Create a sample point cloud near known MOP locations."""
    # Tower site area: scanner at ~(476191, 3636212)
    np.random.seed(42)
    n_points = 1000

    # Create points in a fan shape from scanner position
    scanner_x, scanner_y = 476191.0, 3636212.0
    angles = np.random.uniform(-0.3, 0.3, n_points)  # radians
    distances = np.random.uniform(10, 80, n_points)

    # Cross-shore direction (roughly west)
    base_angle = np.radians(270)  # west

    X = scanner_x + distances * np.cos(base_angle + angles)
    Y = scanner_y + distances * np.sin(base_angle + angles)

    return X, Y, scanner_x, scanner_y


# =============================================================================
# MopTable Loading Tests
# =============================================================================

def test_mop_table_load_default():
    """Test loading MOP table from default location."""
    table = mop.MopTable.load()
    assert len(table) > 0
    assert table.min_mop >= 1
    assert table.max_mop > table.min_mop


def test_mop_table_load_explicit_path():
    """Test loading MOP table from explicit path."""
    path = PROJECT_ROOT / "mop_data" / "MopTable.csv"
    table = mop.MopTable.load(path)
    assert len(table) > 0


def test_mop_table_load_nonexistent():
    """Test that loading from nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        mop.MopTable.load(Path("/nonexistent/path.csv"))


def test_mop_table_columns(mop_table):
    """Test that MOP table has required columns."""
    required = ['MopNum', 'BackXutm', 'BackYutm', 'OffXutm', 'OffYutm']
    for col in required:
        assert col in mop_table.data.columns


def test_mop_table_mop_numbers(mop_table):
    """Test mop_numbers property returns sorted list."""
    mop_nums = mop_table.mop_numbers
    assert isinstance(mop_nums, list)
    assert mop_nums == sorted(mop_nums)
    assert all(isinstance(n, int) for n in mop_nums)


# =============================================================================
# Integer MOP Retrieval Tests
# =============================================================================

def test_get_mop_valid(mop_table):
    """Test getting a valid integer MOP."""
    # Get first MOP number from table
    mop_num = mop_table.mop_numbers[0]
    transect = mop_table.get_mop(mop_num)

    assert isinstance(transect, mop.MopTransect)
    assert transect.mop_num == mop_num
    assert isinstance(transect.back_x, float)
    assert isinstance(transect.back_y, float)
    assert isinstance(transect.off_x, float)
    assert isinstance(transect.off_y, float)


def test_get_mop_invalid(mop_table):
    """Test that getting invalid MOP raises KeyError."""
    invalid_mop = mop_table.max_mop + 1000
    with pytest.raises(KeyError, match=f"MOP {invalid_mop} not found"):
        mop_table.get_mop(invalid_mop)


def test_get_mop_endpoints_match_csv(mop_table):
    """Test that MOP endpoints match CSV values exactly."""
    # Get first row of data
    first_row = mop_table.data.iloc[0]
    mop_num = int(first_row['MopNum'])

    transect = mop_table.get_mop(mop_num)

    assert transect.back_x == pytest.approx(first_row['BackXutm'])
    assert transect.back_y == pytest.approx(first_row['BackYutm'])
    assert transect.off_x == pytest.approx(first_row['OffXutm'])
    assert transect.off_y == pytest.approx(first_row['OffYutm'])


# =============================================================================
# Fractional MOP Tests
# =============================================================================

def test_get_fractional_mop_interpolation(mop_table):
    """Test fractional MOP interpolation between adjacent MOPs."""
    # Use two adjacent MOPs that exist
    mop_nums = mop_table.mop_numbers
    base_mop = mop_nums[len(mop_nums) // 2]  # Pick one from middle
    next_mop = base_mop + 1

    # Skip if next_mop doesn't exist
    if next_mop not in mop_nums:
        pytest.skip("Adjacent MOP not found for interpolation test")

    mop1 = mop_table.get_mop(base_mop)
    mop2 = mop_table.get_mop(next_mop)
    frac = 0.3

    interpolated = mop_table.get_fractional_mop(base_mop + frac)

    # Check interpolation
    expected_back_x = mop1.back_x + frac * (mop2.back_x - mop1.back_x)
    expected_back_y = mop1.back_y + frac * (mop2.back_y - mop1.back_y)
    expected_off_x = mop1.off_x + frac * (mop2.off_x - mop1.off_x)
    expected_off_y = mop1.off_y + frac * (mop2.off_y - mop1.off_y)

    assert interpolated.mop_num == pytest.approx(base_mop + frac)
    assert interpolated.back_x == pytest.approx(expected_back_x)
    assert interpolated.back_y == pytest.approx(expected_back_y)
    assert interpolated.off_x == pytest.approx(expected_off_x)
    assert interpolated.off_y == pytest.approx(expected_off_y)


def test_get_fractional_mop_midpoint(mop_table):
    """Test fractional MOP at 0.5 is exact midpoint."""
    mop_nums = mop_table.mop_numbers
    base_mop = mop_nums[len(mop_nums) // 2]
    next_mop = base_mop + 1

    if next_mop not in mop_nums:
        pytest.skip("Adjacent MOP not found")

    mop1 = mop_table.get_mop(base_mop)
    mop2 = mop_table.get_mop(next_mop)

    interpolated = mop_table.get_fractional_mop(base_mop + 0.5)

    # Midpoint should be average
    assert interpolated.back_x == pytest.approx((mop1.back_x + mop2.back_x) / 2)
    assert interpolated.back_y == pytest.approx((mop1.back_y + mop2.back_y) / 2)


def test_get_fractional_mop_integer_returns_exact(mop_table):
    """Test that fractional MOP with integer value returns exact MOP."""
    mop_num = mop_table.mop_numbers[0]
    fractional = mop_table.get_fractional_mop(float(mop_num))
    exact = mop_table.get_mop(mop_num)

    assert fractional.back_x == exact.back_x
    assert fractional.back_y == exact.back_y
    assert fractional.off_x == exact.off_x
    assert fractional.off_y == exact.off_y


def test_get_fractional_mop_out_of_range(mop_table):
    """Test that out-of-range fractional MOP raises ValueError."""
    with pytest.raises(ValueError, match="out of valid range"):
        mop_table.get_fractional_mop(mop_table.max_mop + 0.5)


# =============================================================================
# MopTransect Properties Tests
# =============================================================================

def test_mop_transect_length(mop_table):
    """Test MopTransect length calculation."""
    transect = mop_table.get_mop(mop_table.mop_numbers[0])

    expected_length = np.sqrt(
        (transect.off_x - transect.back_x)**2 +
        (transect.off_y - transect.back_y)**2
    )

    assert transect.length == pytest.approx(expected_length)
    assert transect.length > 0


def test_mop_transect_azimuth(mop_table):
    """Test MopTransect azimuth calculation."""
    transect = mop_table.get_mop(mop_table.mop_numbers[0])

    assert 0 <= transect.azimuth < 360


def test_mop_transect_to_transect_config(mop_table):
    """Test conversion to TransectConfig."""
    transect = mop_table.get_mop(mop_table.mop_numbers[0])
    config = transect.to_transect_config(
        tolerance=3.0,
        resolution=0.15,
        expansion_rate=0.02,
    )

    assert isinstance(config, profiles.TransectConfig)
    assert config.x1 == transect.back_x
    assert config.y1 == transect.back_y
    assert config.x2 == transect.off_x
    assert config.y2 == transect.off_y
    assert config.tolerance == 3.0
    assert config.resolution == 0.15
    assert config.expansion_rate == 0.02


# =============================================================================
# find_mops_in_bounds Tests
# =============================================================================

def test_find_mops_in_bounds(mop_table):
    """Test finding MOPs within UTM bounds."""
    # Use bounds that should contain some MOPs (California coast)
    # Tower site area
    x_min, x_max = 475000, 478000
    y_min, y_max = 3635000, 3638000

    mops = mop_table.find_mops_in_bounds(x_min, x_max, y_min, y_max)

    # Should find at least some MOPs in this area
    assert isinstance(mops, list)
    # List should be sorted
    assert mops == sorted(mops)


def test_find_mops_in_bounds_empty():
    """Test finding MOPs in area with no MOPs."""
    table = mop.MopTable.load()
    # Use bounds far from California coast
    mops = table.find_mops_in_bounds(0, 100, 0, 100)
    assert mops == []


# =============================================================================
# MOP Selection Tests
# =============================================================================

def test_select_best_mop_centroid(mop_table, sample_point_cloud):
    """Test MOP selection by centroid method."""
    X, Y, _, _ = sample_point_cloud

    # Should return an integer MOP number
    selected = mop.select_best_mop(X, Y, mop_table, method="centroid")

    assert isinstance(selected, int)
    assert selected in mop_table.mop_numbers


def test_select_best_mop_coverage(mop_table, sample_point_cloud):
    """Test MOP selection by coverage method."""
    X, Y, _, _ = sample_point_cloud

    selected = mop.select_best_mop(
        X, Y, mop_table,
        method="coverage",
        tolerance=50.0
    )

    assert isinstance(selected, int)
    assert selected in mop_table.mop_numbers


def test_select_best_mop_nearest_scanner(mop_table, sample_point_cloud):
    """Test MOP selection by nearest_scanner method."""
    X, Y, scanner_x, scanner_y = sample_point_cloud

    selected = mop.select_best_mop(
        X, Y, mop_table,
        scanner_position=(scanner_x, scanner_y),
        method="nearest_scanner"
    )

    assert isinstance(selected, int)
    assert selected in mop_table.mop_numbers


def test_select_best_mop_nearest_scanner_requires_position(mop_table, sample_point_cloud):
    """Test that nearest_scanner method requires scanner_position."""
    X, Y, _, _ = sample_point_cloud

    with pytest.raises(ValueError, match="Scanner position required"):
        mop.select_best_mop(X, Y, mop_table, method="nearest_scanner")


def test_select_best_mop_invalid_method(mop_table, sample_point_cloud):
    """Test that invalid method raises ValueError."""
    X, Y, _, _ = sample_point_cloud

    with pytest.raises(ValueError, match="Unknown selection method"):
        mop.select_best_mop(X, Y, mop_table, method="invalid")


def test_select_best_mop_empty_points(mop_table):
    """Test that empty points raise ValueError."""
    X = np.array([])
    Y = np.array([])

    with pytest.raises(ValueError, match="No valid points"):
        mop.select_best_mop(X, Y, mop_table)


def test_select_best_mop_nan_handling(mop_table, sample_point_cloud):
    """Test that NaN values are handled correctly."""
    X, Y, _, _ = sample_point_cloud

    # Add some NaN values
    X_with_nan = np.concatenate([X, [np.nan, np.nan, np.nan]])
    Y_with_nan = np.concatenate([Y, [np.nan, np.nan, np.nan]])

    # Should still work
    selected = mop.select_best_mop(X_with_nan, Y_with_nan, mop_table)
    assert isinstance(selected, int)


# =============================================================================
# utm2mop_xshore Tests
# =============================================================================

def test_utm2mop_xshore_with_mop_num(mop_table):
    """Test utm2mop_xshore with specified MOP number."""
    mop_num = mop_table.mop_numbers[0]
    transect = mop_table.get_mop(mop_num)

    # Create points along the transect
    t = np.linspace(0, 1, 10)
    X = transect.back_x + t * (transect.off_x - transect.back_x)
    Y = transect.back_y + t * (transect.off_y - transect.back_y)

    mop_nums, xshore = mop.utm2mop_xshore(X, Y, mop_table, mop_num=mop_num)

    # All points should be assigned to the specified MOP
    assert np.all(mop_nums == mop_num)

    # Cross-shore distances should increase along transect
    assert np.all(np.diff(xshore) > 0)

    # First point should be near 0, last near transect length
    assert xshore[0] == pytest.approx(0, abs=1.0)
    assert xshore[-1] == pytest.approx(transect.length, abs=1.0)


def test_utm2mop_xshore_fractional(mop_table):
    """Test utm2mop_xshore with fractional MOP number."""
    mop_nums_list = mop_table.mop_numbers
    base_mop = mop_nums_list[len(mop_nums_list) // 2]

    if base_mop + 1 not in mop_nums_list:
        pytest.skip("Adjacent MOP not found")

    frac_mop = base_mop + 0.5
    transect = mop_table.get_fractional_mop(frac_mop)

    # Create points along the interpolated transect
    t = np.linspace(0, 1, 5)
    X = transect.back_x + t * (transect.off_x - transect.back_x)
    Y = transect.back_y + t * (transect.off_y - transect.back_y)

    mop_nums, xshore = mop.utm2mop_xshore(X, Y, mop_table, mop_num=frac_mop)

    assert np.all(mop_nums == pytest.approx(frac_mop))


# =============================================================================
# get_mop_transect Convenience Function Tests
# =============================================================================

def test_get_mop_transect_integer():
    """Test get_mop_transect with integer MOP."""
    table = mop.MopTable.load()
    mop_num = table.mop_numbers[0]

    config = mop.get_mop_transect(
        mop_num,
        tolerance=2.5,
        resolution=0.1,
        expansion_rate=0.01,
    )

    assert isinstance(config, profiles.TransectConfig)
    assert config.tolerance == 2.5
    assert config.resolution == 0.1
    assert config.expansion_rate == 0.01


def test_get_mop_transect_fractional():
    """Test get_mop_transect with fractional MOP."""
    table = mop.MopTable.load()
    mop_nums = table.mop_numbers
    base_mop = mop_nums[len(mop_nums) // 2]

    if base_mop + 1 not in mop_nums:
        pytest.skip("Adjacent MOP not found")

    config = mop.get_mop_transect(base_mop + 0.3)

    assert isinstance(config, profiles.TransectConfig)


def test_get_mop_transect_with_table():
    """Test get_mop_transect with pre-loaded table."""
    table = mop.MopTable.load()
    mop_num = table.mop_numbers[0]

    config = mop.get_mop_transect(mop_num, mop_table=table)

    assert isinstance(config, profiles.TransectConfig)


# =============================================================================
# format_mop_filename_suffix Tests
# =============================================================================

def test_format_mop_filename_suffix_integer():
    """Test filename suffix for integer MOP."""
    assert mop.format_mop_filename_suffix(456) == "_MOP456"
    assert mop.format_mop_filename_suffix(123) == "_MOP123"


def test_format_mop_filename_suffix_fractional():
    """Test filename suffix for fractional MOP."""
    assert mop.format_mop_filename_suffix(456.3) == "_MOP456.3"
    assert mop.format_mop_filename_suffix(456.0) == "_MOP456"  # .0 should be treated as integer


def test_format_mop_filename_suffix_none():
    """Test filename suffix for None."""
    assert mop.format_mop_filename_suffix(None) == ""


# =============================================================================
# _distance_point_to_line Tests
# =============================================================================

def test_distance_point_to_line_on_line():
    """Test distance for point on the line."""
    # Horizontal line from (0,0) to (10,0)
    perp, xshore = mop._distance_point_to_line(5, 0, 0, 0, 10, 0)

    assert perp == pytest.approx(0.0)
    assert xshore == pytest.approx(5.0)


def test_distance_point_to_line_perpendicular():
    """Test distance for point perpendicular to line."""
    # Horizontal line from (0,0) to (10,0)
    # Point at (5, 3)
    perp, xshore = mop._distance_point_to_line(5, 3, 0, 0, 10, 0)

    assert perp == pytest.approx(3.0)
    assert xshore == pytest.approx(5.0)


def test_distance_point_to_line_diagonal():
    """Test distance for diagonal line."""
    # Line from (0,0) to (10,10)
    # Point at origin should have 0 distance
    perp, xshore = mop._distance_point_to_line(0, 0, 0, 0, 10, 10)

    assert perp == pytest.approx(0.0)
    assert xshore == pytest.approx(0.0)


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_mop_workflow(mop_table, sample_point_cloud):
    """Test complete workflow: select MOP, get transect, convert to config."""
    X, Y, scanner_x, scanner_y = sample_point_cloud

    # Select best MOP
    selected_mop = mop.select_best_mop(
        X, Y, mop_table,
        scanner_position=(scanner_x, scanner_y),
        method="centroid"
    )

    # Get transect
    transect = mop_table.get_mop(selected_mop)

    # Convert to TransectConfig
    config = transect.to_transect_config(
        tolerance=2.0,
        resolution=0.1,
        expansion_rate=0.02,
    )

    # Verify config is valid for profile extraction
    assert config.x1 != config.x2 or config.y1 != config.y2  # Not a point
    assert config.tolerance > 0
    assert config.resolution > 0


def test_fractional_mop_workflow():
    """Test workflow with fractional MOP number."""
    table = mop.MopTable.load()
    mop_nums = table.mop_numbers

    # Find two adjacent MOPs
    for i, mop_num in enumerate(mop_nums[:-1]):
        if mop_nums[i+1] == mop_num + 1:
            frac_mop = mop_num + 0.3
            break
    else:
        pytest.skip("No adjacent MOPs found")

    # Get fractional transect
    transect = table.get_fractional_mop(frac_mop)

    # Convert to config
    config = transect.to_transect_config()

    # Verify
    assert isinstance(config, profiles.TransectConfig)
    assert transect.mop_num == frac_mop
