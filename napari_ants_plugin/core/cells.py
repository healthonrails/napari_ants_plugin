import numpy as np
from scipy.spatial import KDTree
import argparse
import glob
import pandas as pd


def remove_duplicate_cells(cell_locations, cell_size_radius=1.0):
    """
    Efficiently removes duplicate cell locations from a list of (z, y, x) coordinates,
    considering a cell size radius.

    This function uses a KD-tree for fast nearest neighbor searches to identify
    and remove duplicate points. If multiple cell locations are within the
    specified cell_size_radius of each other, only the first encountered point is kept.

    Args:
        cell_locations (list of tuples or numpy.ndarray): A list of cell locations,
            where each location is a tuple or numpy array of the form (z, y, x).
        cell_size_radius (float): The radius within which two cell locations are considered duplicates.
            Must be a non-negative value.

    Returns:
        list of tuples: A list of unique cell locations (tuples) after removing duplicates.
    """
    # Input validation
    if not isinstance(cell_locations, (list, np.ndarray)):
        raise TypeError("cell_locations must be a list or numpy array.")
    if not isinstance(cell_size_radius, (int, float)):
        raise TypeError("cell_size_radius must be a number.")
    if cell_size_radius < 0:
        raise ValueError("cell_size_radius must be non-negative.")

    if len(cell_locations) == 0:
        return []

    try:
        cell_locations_np = np.array(cell_locations, dtype=np.float64)
        if cell_locations_np.ndim == 1 and len(cell_locations_np) == 3:
            cell_locations_np = cell_locations_np.reshape(1, -1)
        elif cell_locations_np.ndim != 2 or cell_locations_np.shape[1] != 3:
            raise ValueError(
                "cell_locations must be a list of tuples/lists/numpy arrays of length 3 (z, y, x).")
    except ValueError as e:
        raise ValueError(f"Invalid format for cell_locations: {e}")
    except TypeError:
        raise TypeError("cell_locations must contain numerical coordinates.")

    if cell_locations_np.shape[0] <= 1:
        return [tuple(loc) for loc in cell_locations_np.tolist()]

    kdtree = KDTree(cell_locations_np)
    unique_cells = []
    processed_indices = set()

    for i in range(cell_locations_np.shape[0]):
        if i in processed_indices:
            continue

        nearby_indices = kdtree.query_ball_point(
            cell_locations_np[i], r=cell_size_radius)
        unique_cells.append(tuple(cell_locations_np[i].tolist()))
        processed_indices.update(nearby_indices)

    return unique_cells


def load_locations_from_csv(csv_filepath):
    """
    Loads cell locations from a CSV file using pandas.

    Assumes the CSV file contains a header with columns: 'z', 'y', 'x' and numerical data.

    Args:
        csv_filepath (str): Path to the CSV file.

    Returns:
        list of tuples: List of cell locations as tuples (z, y, x). If the file doesn't have the expected
                        columns, an empty list is returned with a warning.
    """
    try:
        df = pd.read_csv(csv_filepath)
        expected_cols = ['z', 'y', 'x']
        if not all(col in df.columns for col in expected_cols):
            print(
                f"Warning: File {csv_filepath} does not have the expected columns {expected_cols}. Skipping.")
            return []
        # Ensure the columns are in the correct order (if not already)
        df = df[expected_cols]
        return list(df.itertuples(index=False, name=None))
    except Exception as e:
        print(f"Error reading {csv_filepath}: {e}")
        return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge CSV files with cell locations (header: z,y,x), remove duplicates, and output unique points."
    )
    parser.add_argument('--input_pattern', '-i', type=str, default='detected_cells_*.csv',
                        help='Glob pattern for input CSV files (default: detected_cells_*.csv)')
    parser.add_argument('--radius', '-r', type=float, default=5.0,
                        help='Cell size radius for considering duplicates (default: 5.0)')
    parser.add_argument('--output_csv', '-o', type=str, default='points.csv',
                        help='Output CSV file to save unique cell locations (default: points.csv)')
    args = parser.parse_args()

    # Find CSV files matching the provided glob pattern
    input_files = glob.glob(args.input_pattern)
    if not input_files:
        print(f"No files found matching pattern: {args.input_pattern}")
        exit(1)

    all_locations = []
    print("Loading cell locations from:")
    for file in input_files:
        print(f"  {file}")
        locations = load_locations_from_csv(file)
        all_locations.extend(locations)

    if not all_locations:
        print("No valid cell locations were found in the provided CSV files.")
        exit(1)

    unique_locations = remove_duplicate_cells(all_locations, args.radius)
    print(f"\nTotal locations loaded: {len(all_locations)}")
    print(f"Unique locations after deduplication: {len(unique_locations)}")

    try:
        # Create a DataFrame with the proper header and save to CSV
        df_out = pd.DataFrame(unique_locations, columns=['z', 'y', 'x'])
        df_out.to_csv(args.output_csv, index=False)
        print(f"Unique cell locations saved to: {args.output_csv}")
    except Exception as e:
        print(f"Error writing to output CSV file: {e}")
