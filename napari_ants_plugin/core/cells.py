import numpy as np
from scipy.spatial import KDTree
import argparse
import csv


def remove_duplicate_cells(cell_locations, cell_size_radius=1.0):
    """
    Efficiently removes duplicate cell locations from a list of (z, y, x) coordinates,
    considering a cell size radius.

    This function uses a KD-tree for fast nearest neighbor searches to identify
    and remove duplicate points. If multiple cell locations are within the
    specified cell_size_radius of each other, only the first encountered
    location is kept.

    Args:
        cell_locations (list of tuples or numpy.ndarray): A list of cell locations,
            where each location is a tuple or numpy array of the form (z, y, x).
            e.g., [(z1, y1, x1), (z2, y2, x2), ...] or np.array([[z1, y1, z1], [z2, y2, z2], ...])
        cell_size_radius (float): The radius within which two cell locations are
            considered duplicates.  Units should be consistent with the coordinates.
            Must be a non-negative value.

    Returns:
        list of tuples: A list of unique cell locations (tuples) after removing duplicates.
                       The order of points in the output list is not guaranteed to be the same
                       as the input order, but the first encountered point in each duplicate
                       cluster is preserved.

    Raises:
        TypeError: if cell_locations is not a list or numpy array, or if cell_size_radius is not a number.
        ValueError: if cell_size_radius is negative.
        ValueError: if cell_locations is not a list of tuples/lists/numpy arrays of length 3.

    Example:
        locations = [(1, 2, 3), (1.1, 2.1, 3.2), (4, 5, 6), (1, 2, 3.1), (7, 8, 9)]
        radius = 0.5
        unique_locations = remove_duplicate_cells(locations, radius)
        print(unique_locations) # Expected output might be: [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
                               # (Order might vary, but the representative points will be there)
    """

    # Input validation - crucial for robust functions
    if not isinstance(cell_locations, (list, np.ndarray)):
        raise TypeError("cell_locations must be a list or numpy array.")
    if not isinstance(cell_size_radius, (int, float)):
        raise TypeError("cell_size_radius must be a number.")
    if cell_size_radius < 0:
        raise ValueError("cell_size_radius must be non-negative.")

    if not cell_locations:  # Handle empty input list efficiently
        return []

    # Convert input to numpy array for KDTree operations, ensuring correct shape and type
    try:
        # Use float64 for precision
        cell_locations_np = np.array(cell_locations, dtype=np.float64)
        # Handle single point input as list/tuple
        if cell_locations_np.ndim == 1 and len(cell_locations_np) == 3:
            cell_locations_np = cell_locations_np.reshape(
                1, -1)  # Reshape to 2D array
        elif cell_locations_np.ndim != 2 or cell_locations_np.shape[1] != 3:
            raise ValueError(
                "cell_locations must be a list of tuples/lists/numpy arrays of length 3 (z, y, x).")
    except ValueError as e:
        raise ValueError(f"Invalid format for cell_locations: {e}")
    except TypeError:  # Catch potential type errors during numpy conversion
        raise TypeError("cell_locations must contain numerical coordinates.")

    # Optimization: No duplicates possible with 0 or 1 point
    if cell_locations_np.shape[0] <= 1:
        # Return as tuples, consistent output
        return [tuple(loc) for loc in cell_locations_np.tolist()]

    # Build KDTree for efficient nearest neighbor search
    kdtree = KDTree(cell_locations_np)

    unique_cells = []
    processed_indices = set()  # Keep track of indices already considered as duplicates

    for i in range(cell_locations_np.shape[0]):
        if i in processed_indices:  # Skip if already marked as duplicate
            continue

        # Find indices of points within cell_size_radius of the current point
        # query_ball_point is efficient for finding points within a radius
        nearby_indices = kdtree.query_ball_point(
            cell_locations_np[i], r=cell_size_radius)

        # Add the current cell location as it's the representative of this cluster
        # Convert back to tuple for list of tuples output
        unique_cells.append(tuple(cell_locations_np[i].tolist()))

        # Mark all nearby indices as processed to avoid re-processing them
        processed_indices.update(nearby_indices)

    return unique_cells


def load_locations_from_csv(csv_filepath):
    """
    Loads cell locations from a CSV file.

    Assumes the CSV file contains comma-separated (z, y, x) coordinates, one per row.

    Args:
        csv_filepath (str): Path to the CSV file.

    Returns:
        list of tuples: List of cell locations as tuples (z, y, x).
    """
    locations = []
    try:
        with open(csv_filepath, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                try:
                    z, y, x = map(float, row)  # Convert each element to float
                    locations.append((z, y, x))
                except ValueError:
                    print(
                        f"Warning: Skipping invalid row in CSV: {row}.  Expected numerical (z, y, x) values.")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}")
    return locations


if __name__ == '__main__':
    # Argument Parser setup
    parser = argparse.ArgumentParser(
        description='Remove duplicate cell locations based on cell size radius.')

    print(remove_duplicate_cells.__doc__)  # Print function docstring

    parser.add_argument('--input_csv', '-i', type=str,
                        help='Path to the input CSV file containing cell locations (z, y, x) per row.')
    parser.add_argument('--radius', '-r', type=float, default=5.0,
                        # Default radius is now 5.0
                        help='Cell size radius for considering duplicates (default: 5.0).')
    parser.add_argument('--output_csv', '-o', type=str,
                        help='Path to save the unique cell locations to a CSV file.')
    args = parser.parse_args()

    cell_locations = []

    if args.input_csv:
        try:
            cell_locations = load_locations_from_csv(args.input_csv)
            if not cell_locations:
                print("Warning: No valid cell locations found in the CSV file.")
            else:
                print(
                    f"Loaded {len(cell_locations)} locations from CSV: {args.input_csv}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            exit(1)  # Exit with error code
    else:
        # Default locations if no CSV provided for demonstration
        cell_locations = [(1, 2, 3), (1.1, 2.1, 3.2),
                          (4, 5, 6), (1, 2, 3.1), (7, 8, 9)]
        print("Using default cell locations for demonstration.")

    cell_size_radius = args.radius

    if cell_locations:  # Only process if there are locations to process
        unique_locations = remove_duplicate_cells(
            cell_locations, cell_size_radius)
        print(f"Cell Size Radius: {cell_size_radius}")
        print(f"Number of Unique Locations: {len(unique_locations)}")

        if args.output_csv:
            output_csv_filepath = args.output_csv
            try:
                with open(output_csv_filepath, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    # Write list of tuples directly
                    csv_writer.writerows(unique_locations)
                print(f"Unique cell locations saved to: {output_csv_filepath}")
            except Exception as e:
                print(f"Error writing to output CSV file: {e}")
        else:
            # Print only first 10 unique locations to console for brevity if many
            print(
                f"Unique Locations (first 10): {unique_locations[:min(10, len(unique_locations))]}")
            print(
                "To save unique locations to a CSV file, use the --output_csv or -o argument.")

    else:
        print("No cell locations to process.")

    # Example Usage/Test cases are now moved to be part of the argparse execution if no CSV is given.
    if not args.input_csv:
        print("\n--- Running Test Cases (No CSV Input) ---")
        # Test cases to demonstrate functionality and edge cases
        locations1 = [(1, 2, 3), (1.1, 2.1, 3.2),
                      (4, 5, 6), (1, 2, 3.1), (7, 8, 9)]
        radius1 = 0.5
        unique_locations1 = remove_duplicate_cells(locations1, radius1)
        print(
            f"\nTest 1 - Locations: {locations1}, Radius: {radius1}, Unique Locations: {unique_locations1}")

        locations2 = []  # Empty list
        radius2 = 0.1
        unique_locations2 = remove_duplicate_cells(locations2, radius2)
        print(
            f"Test 2 - Locations: {locations2}, Radius: {radius2}, Unique Locations: {unique_locations2}")

        locations3 = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]  # All duplicates
        radius3 = 0.01
        unique_locations3 = remove_duplicate_cells(locations3, radius3)
        print(
            f"Test 3 - Locations: {locations3}, Radius: {radius3}, Unique Locations: {unique_locations3}")

        locations4 = [(10, 20, 30)]  # Single location
        radius4 = 1.0
        unique_locations4 = remove_duplicate_cells(locations4, radius4)
        print(
            f"Test 4 - Locations: {locations4}, Radius: {radius4}, Unique Locations: {unique_locations4}")

        locations5 = [(2, 2, 2), (5, 5, 5), (2.05, 2.05, 2.05),
                      (8, 8, 8), (5.1, 5.1, 5.1)]
        radius5 = 0.2
        unique_locations5 = remove_duplicate_cells(locations5, radius5)
        print(
            f"Test 5 - Locations: {locations5}, Radius: {radius5}, Unique Locations: {unique_locations5}")
