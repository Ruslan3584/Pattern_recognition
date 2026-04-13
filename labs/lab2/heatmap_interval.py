import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Minimize the risk of the Bayesian strategy for interval loss function."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number probability values. [0, 250]"
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.n <= 0 or args.n > 250:
        raise Exception("n must be in range [0, 250]")

    # Set a random seed
    np.random.seed(args.seed)

    # Generate a random heatmap
    heatmap = np.random.randint(low=0, high=250, size=args.n).astype(np.float32)
    heatmap_norm = heatmap / np.sum(heatmap)

    cumsum = np.cumsum(heatmap_norm)
    idx = np.argmax(cumsum >= 0.5)
    res = idx - 1 if cumsum[idx] >= 0.5 else None

    print("Input heatmap: ", heatmap)
    print("Heatmap normalized: ", heatmap_norm)
    print("Result:", res)


if __name__ == "__main__":
    main()
