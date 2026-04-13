import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ---------------------- Core utilities ----------------------


def normalize(hist: np.ndarray) -> np.ndarray:
    total = np.sum(hist)
    if total <= 0:
        raise ValueError("Histogram sum must be positive")
    return hist / total


def sample_from_histogram(probabilities: np.ndarray) -> int:
    x = np.random.rand()
    cum = np.cumsum(probabilities)
    return int(np.searchsorted(cum, x, side="left"))


def generate_histogram(n: int) -> np.ndarray:
    hist = np.random.randint(0, 100, n)
    return normalize(hist)


# ---------------------- Strategies ----------------------


def quadratic(values: np.ndarray, hist: np.ndarray) -> int:
    mean = np.dot(values, hist)  # already normalized
    return values[np.argmin((values - mean) ** 2)]


def binary(values: np.ndarray, hist: np.ndarray) -> int:
    return values[np.argmax(hist)]


def third(values: np.ndarray, hist: np.ndarray, alpha: float = 0.0) -> int:
    """
    Combined objective:
        quadratic risk - alpha * p(k')
    """
    mean = np.dot(values, hist)

    # quadratic risk term (vectorized)
    quad = (values - mean) ** 2

    # equivalent to original formulation
    scores = quad - alpha * hist

    return values[np.argmin(scores)]


# ---------------------- Risk functions ----------------------


def r_binary(q: int, values: np.ndarray, hist: np.ndarray) -> float:
    return 1.0 - hist[np.where(values == q)[0][0]]


def r_quadratic(q: int, values: np.ndarray, hist: np.ndarray) -> float:
    return np.sum(hist * (values - q) ** 2)


# ---------------------- Monte Carlo evaluation ----------------------


def evaluate_strategies(values: np.ndarray, n_samples: int = 10_000):
    quad_risks = np.empty(n_samples)
    bin_risks = np.empty(n_samples)

    for i in range(n_samples):
        hist = generate_histogram(len(values))

        q_q = quadratic(values, hist)
        q_b = binary(values, hist)

        quad_risks[i] = r_quadratic(q_q, values, hist)
        bin_risks[i] = r_binary(q_b, values, hist)

    return np.mean(quad_risks), np.mean(bin_risks)


def evaluate_third(values: np.ndarray, alpha: float, n_samples: int = 10_000):
    quad_risks = np.empty(n_samples)
    bin_risks = np.empty(n_samples)

    for i in range(n_samples):
        hist = generate_histogram(len(values))
        q = third(values, hist, alpha)

        quad_risks[i] = r_quadratic(q, values, hist)
        bin_risks[i] = r_binary(q, values, hist)

    return np.mean(bin_risks), np.mean(quad_risks)


def sweep_alpha(values: np.ndarray, alphas: np.ndarray, n_samples: int):
    Rb = np.empty_like(alphas, dtype=float)
    Rq = np.empty_like(alphas, dtype=float)

    for i, a in enumerate(tqdm(alphas)):
        Rb[i], Rq[i] = evaluate_third(values, a, n_samples)

    return Rb, Rq


def plot_curve(x, y, ylabel):
    plt.plot(x, y)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(ylabel)
    plt.savefig(f"alpha_{ylabel.replace(' ', '_')}.png")
    plt.close()


def main():
    """Non-Bayesian strategy analysis."""
    parser = argparse.ArgumentParser(description="Non-Bayesian strategy analysis.")
    parser.add_argument("--values", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=10_000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    values = np.array(args.values, dtype=np.float64)

    # Single histogram demo
    hist = generate_histogram(len(values))

    q_q = quadratic(values, hist)
    q_b = binary(values, hist)
    q_t = third(values, hist)

    print("q_quadratic:", q_q)
    print("q_binary:", q_b)
    print("q_third:", q_t)

    print("R_binary:", r_binary(q_b, values, hist))
    print("R_quadratic:", r_quadratic(q_q, values, hist))

    # Monte Carlo
    mean_q, mean_b = evaluate_strategies(values, args.samples)
    print("Mean quadratic risk:", mean_q)
    print("Mean binary risk:", mean_b)

    # Alpha sweep
    alpha = np.linspace(-50, 50, 15)
    Rb, Rq = sweep_alpha(values, alpha, args.samples)

    plot_curve(alpha, Rq, r"R_quadratic")
    plot_curve(alpha, Rb, r"R_binary")


if __name__ == "__main__":
    main()
