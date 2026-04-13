import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

N = 10**6
N_RANGES = 10_000
N_REPEATS = 200  # reduce noise, not too large
OUTPUT_PREFIX = "benchmark"


def partial_sum(cumsum: np.ndarray, m: int, n: int) -> int:
    """
    Calculate a partial sum of an array from m-index to n-index.
    Inclusion-Exclusion principle.

    Args:
        cumsum: Cumulative sum of an array.
        m: Starting index.
        n: Ending index.
    """
    if m >= n:
        raise Exception("ERROR, wrong parameters: m >= n ")
    return cumsum[n] - cumsum[m]


def simple_partial_sum(array: np.ndarray, m: int, n: int):
    """Calculate a partial sum using the standard np.sum function.

    Args:
        array: Input array.
        m: Starting index.
        n: Ending index.
    """
    if m >= n:
        raise Exception("ERROR, wrong parameters: m >= n ")
    return np.sum(array[m:n])


def summed_area_table(cumsum: np.ndarray, m: int, n: int, i: int, j: int) -> int:
    """Calculate the summed area table of a 2D array [m:n,i:j].
    Inclusion-Exclusion principle.

    Args:
        cumsum: Cumulative sum of an array.
        m: Starting index on axis 0.
        n: Ending index on axis 0.
        i: Starting index on axis 1.
        j: Ending index on axis 1.

    """
    if m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: m >= n or i >= j ")

    so = cumsum[n, j]
    s1 = -cumsum[n, i]
    s2 = -cumsum[m, j]
    s3 = cumsum[m, i]

    return so + s1 + s2 + s3


def simple_summed_area_table(array, m, n, i, j) -> int:
    """Calculate a summed area table using the standard np.sum function.

    Args:
        array: Input array.
        m: Starting index on axis 0.
        n: Ending index on axis 0.
        i: Starting index on axis 1.
        j: Ending index on axis 1.

    """
    if m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: m >= n or i >= j ")
    return np.sum(array[m:n, i:j])


def summed_volume_table(
    cumsum: np.ndarray, l: int, k: int, m: int, n: int, i: int, j: int
) -> int:
    """Calculate the summed volume table of a 2D array [l:k, m:n,i:j].
    Inclusion-Exclusion principle.

    Args:
        cumsum: Cumulative sum of an array.
        l: Starting index on axis 0.
        k: Ending index on axis 0.
        m: Starting index on axis 1.
        n: Ending index on axis 1.
        i: Starting index on axis 2.
        j: Ending index on axis 2.
    """

    if l >= k or m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: l >=k or m >= n or i >= j ")

    s0 = cumsum[k, n, j]
    s1 = -cumsum[k, n, i]
    s2 = -cumsum[k, m, j]
    s3 = -cumsum[l, n, j]
    s4 = cumsum[l, m, j]
    s5 = cumsum[l, n, i]
    s6 = cumsum[k, m, i]
    s7 = -cumsum[l, m, i]

    return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7


def simple_summed_volume_table(
    array: np.ndarray, l: int, k: int, m: int, n: int, i: int, j: int
):
    """Calculate a volume area table using the standard np.sum function.

    Args:
        array: Input array.
        l: Starting index on axis 0.
        k: Ending index on axis 0.
        m: Starting index on axis 1.
        n: Ending index on axis 1.
        i: Starting index on axis 2.
        j: Ending index on axis 2.

    """
    if l >= k or m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: l >=k or m >= n or i >= j ")

    return np.sum(array[l:k, m:n, i:j])


def adding_zeros(cumsum: np.ndarray) -> np.ndarray:
    """
    Add zero rows and columns to cumsum to exclude special cases( for example: m == 0, n == 0, and so on).
    """
    return np.pad(cumsum, (1, 0), "constant", constant_values=0)


def generate_ranges(n_ranges, max_start, lengths):
    """Vectorized range generation."""
    ranges = []
    for L in lengths:
        starts = np.random.randint(0, max_start - L - 1, size=n_ranges)
        ends = starts + L
        ranges.append(np.stack([starts, ends], axis=1))
    return ranges


def benchmark(func, data, ranges_per_length, lengths):
    means = []
    stds = []

    for i in tqdm(range(len(lengths)), desc=f"Benchmarking {func.__name__}"):
        ranges = ranges_per_length[i]
        times = []

        for _ in range(N_REPEATS):
            start = time.perf_counter()

            for m, n in ranges:
                func(data, m, n)

            end = time.perf_counter()
            times.append(end - start)

        means.append(np.mean(times))
        stds.append(np.std(times))

    return np.array(means), np.array(stds)


def save_plot(x, y, yerr, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, "o-", label=title)
    plt.errorbar(x, y, yerr=yerr, capsize=3)
    plt.xlabel("Segment length")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    example_1 = np.random.randint(1, 50, (100))
    cum_example_1 = adding_zeros(example_1.cumsum(axis=0))

    example_2 = np.random.randint(1, 50, (7, 7))
    cum_example_2 = adding_zeros(example_2.cumsum(axis=0).cumsum(axis=1))

    example_3 = np.random.randint(1, 50, (5, 7, 8))
    cum_example_3 = adding_zeros(example_3.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2))

    one_dim = partial_sum(cum_example_1, 2, 50)
    print("1d array: ", one_dim)
    print("m = 2, n = 50:")
    print("Sum for 1d array: ", one_dim)
    print("-" * 20)

    two_dim = summed_area_table(cum_example_2, 0, 5, 1, 3)
    print("2d array: ", example_2)
    print("m = 0, n = 5, i = 1, j = 3:")
    print("Summed area table for 2d array: ", two_dim)
    print("-" * 20)

    three_dim = summed_volume_table(cum_example_3, 0, 5, 0, 5, 1, 4)
    print("3d array: ", example_3)
    print("l = 0, k = 5, m = 0, n = 5, i = 1, j = 4:")
    print("Summed volume table for 3d array: ", three_dim)
    print("-" * 20)

    # ---------------------- Run benchmarks ----------------------

    np.random.seed(42)

    array = np.random.randint(0, 10**6, N)
    cum_array = adding_zeros(array.cumsum(axis=0))

    lengths = np.arange(100, 10_000, 200)

    ranges_per_length = generate_ranges(N_RANGES, len(array), lengths)

    mean_fast, std_fast = benchmark(partial_sum, cum_array, ranges_per_length, lengths)
    mean_np, std_np = benchmark(simple_partial_sum, array, ranges_per_length, lengths)

    save_plot(
        lengths, mean_fast, std_fast, "Prefix Sum (O(1))", f"{OUTPUT_PREFIX}_prefix.png"
    )

    save_plot(
        lengths, mean_np, std_np, "Naive NumPy sum (O(n))", f"{OUTPUT_PREFIX}_numpy.png"
    )


if __name__ == "__main__":
    main()
