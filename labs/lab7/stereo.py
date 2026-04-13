import numpy as np
from PIL import Image
import argparse
import time
from multiprocessing import Pool, cpu_count


def compute_row_disparity(args):
    """
    Processes a single row using Dynamic Programming.
    """
    row_idx, left_row, right_row, max_disparity, alpha = args
    width = left_row.shape[0]

    # Cost Volume for this row: [width, max_disparity + 1]
    # init with a large value
    cost_vol = np.full((width, max_disparity + 1), fill_value=1e9, dtype=np.float32)
    # To reconstruct the path
    parent_idx = np.zeros((width, max_disparity + 1), dtype=np.int32)

    # Initial costs for the first pixel
    for d in range(max_disparity + 1):
        if 0 <= 0 - d < width:
            cost_vol[0, d] = abs(left_row[0] - right_row[0 - d])
        else:
            cost_vol[0, d] = 255.0  # Penalty for out of bounds

    # Forward Pass
    # Precompute the smoothness penalty matrix for broadcasting
    indices = np.arange(max_disparity + 1)
    smoothness_penalty = alpha * np.abs(indices[:, None] - indices)

    for x in range(1, width):
        # Pixel matching cost for all possible disparities at this pixel
        pixel_costs = np.array([
            abs(left_row[x] - right_row[x - d]) if x - d >= 0 else 255.0
            for d in range(max_disparity + 1)
        ])

        # For each possible disparity 'd' at current pixel x,
        # find min(previous_cost[d_prev] + smoothness(d, d_prev))
        prev_costs_plus_penalty = cost_vol[x - x % 1 - 1] + smoothness_penalty

        min_prev_indices = np.argmin(prev_costs_plus_penalty, axis=1)
        min_prev_costs = prev_costs_plus_penalty[np.arange(max_disparity + 1), min_prev_indices]

        cost_vol[x] = pixel_costs + min_prev_costs
        parent_idx[x] = min_prev_indices

    # Backward Pass (Backtracking)
    disparities = np.zeros(width, dtype=np.uint8)
    curr_d = np.argmin(cost_vol[-1])

    for x in range(width - 1, -1, -1):
        disparities[x] = curr_d
        curr_d = parent_idx[x, curr_d]

    return disparities


def main():
    parser = argparse.ArgumentParser(description="Stereo DP Algorithm.")
    parser.add_argument("--left", type=str, required=True, help="Path to left image")
    parser.add_argument("--right", type=str, required=True, help="Path to right image")
    parser.add_argument("--alpha", type=float, default=1.0, help="Smoothness weight")
    parser.add_argument("--maxd", type=int, default=30, help="Max disparity")
    args = parser.parse_args()

    # Load and convert to grayscale
    left_img = np.array(Image.open(args.left).convert("L"), dtype=np.float32)
    right_img = np.array(Image.open(args.right).convert("L"), dtype=np.float32)

    height, width = left_img.shape
    start_time = time.perf_counter()

    # Prepare arguments for multiprocessing
    row_args = [
        (y, left_img[y], right_img[y], args.maxd, args.alpha)
        for y in range(height)
    ]

    print(f"Processing {height} rows using {cpu_count()} cores...")

    # Execute in parallel
    with Pool(cpu_count()) as pool:
        disparity_map = pool.map(compute_row_disparity, row_args)

    disparity_map = np.array(disparity_map)

    # Normalization: Map [0, maxd] to [0, 255] for visualization
    # We use the actual max found to preserve contrast
    d_min, d_max = disparity_map.min(), disparity_map.max()
    if d_max > d_min:
        norm_map = (disparity_map - d_min) * (255.0 / (d_max - d_min))
    else:
        norm_map = disparity_map

    norm_map = norm_map.astype(np.uint8)

    # Save output
    Image.fromarray(norm_map).save("disparity_map.png")

    end_time = time.perf_counter()
    print(f"Done! Execution time: {end_time - start_time:.2f} seconds")
    print("Result saved as disparity_map.png")


if __name__ == "__main__":
    main()