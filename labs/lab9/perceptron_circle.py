import numpy as np
import argparse
import matplotlib.pyplot as plt


def split_data(data: np.ndarray, a: float, b: float, r: float):
    if r <= 0:
        raise ValueError("Radius must be positive")

    xy = data[:, 2:4]
    dist_sq = np.sum((xy - np.array([a, b])) ** 2, axis=1)

    mask_in = dist_sq < r * r
    mask_out = ~mask_in

    if not np.any(mask_in):
        raise ValueError("No points inside the circle")

    plot_points = {
        "in": xy[mask_in],
        "out": xy[mask_out],
    }

    # Labeling: inside = -1, outside = +1
    labels = np.where(mask_in, -1.0, 1.0)

    return data, labels, plot_points


def perceptron(X: np.ndarray, y: np.ndarray, max_iter: int = 10_000):
    """
    Vectorized perceptron with early stopping.
    """
    w = np.zeros(X.shape[1])

    for _ in range(max_iter):
        margins = y * (X @ w)
        misclassified = margins <= 0

        if not np.any(misclassified):
            break

        # Batch update (faster convergence than single-step loop)
        w += np.sum((y[misclassified, None] * X[misclassified]), axis=0)

    else:
        raise RuntimeError("Perceptron did not converge")

    # Normalize safely
    if abs(w[1]) < 1e-12:
        raise RuntimeError("Degenerate solution (w[1] ≈ 0)")

    w /= w[1]

    # Extract circle parameters
    a = -0.5 * w[2]
    b = -0.5 * w[3]
    r_sq = a * a + b * b - w[0]
    r = np.sqrt(max(r_sq, 0.0))

    return a, b, r


def visualize_results(plot_points, original, result):
    a, b, r = original
    a_t, b_t, r_t = result

    plt.figure(figsize=(8, 8))

    plt.scatter(*plot_points["in"].T, alpha=0.6, label="Inside")
    plt.scatter(*plot_points["out"].T, alpha=0.6, label="Outside")

    theta = np.linspace(0, 2 * np.pi, 200)

    plt.plot(a + r * np.cos(theta), b + r * np.sin(theta), "--", label="Original")
    plt.plot(a_t + r_t * np.cos(theta), b_t + r_t * np.sin(theta), label="Predicted")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Perceptron Circle Fit")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Perceptron Circle Fit.")
    parser.add_argument("--n", type=int, default=100, help="Number of points to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    a = rng.integers(-10, 10)
    b = rng.integers(-10, 10)
    r = rng.integers(5, 15)

    points = rng.uniform(-25, 25, (args.n, 2))

    X = np.column_stack((
        np.ones(args.n),
        np.sum(points ** 2, axis=1),
        points
    ))

    X, y, plot_points = split_data(X, a, b, r)

    result = perceptron(X, y)

    print(f"Original:  a={a}, b={b}, r={r}")
    print(f"Predicted: a={result[0]:.3f}, b={result[1]:.3f}, r={result[2]:.3f}")

    visualize_results(plot_points, (a, b, r), result)


if __name__ == "__main__":
    main()