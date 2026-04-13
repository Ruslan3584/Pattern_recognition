import argparse
from time import perf_counter
import numpy as np
from sklearn.datasets import fetch_openml
from tqdm import tqdm

def get_apriori_prob(resp):
    """E-step helper: Calculates P(k) based on responsibilities."""
    return resp.mean(axis=0)

def update_pixel_probs(images, resp, apr_p):
    """M-step: Calculates P(pixel=1 | cluster) using vectorized weighted averages."""
    N, H, W = images.shape
    flat_images = images.reshape(N, -1)

    # Weighted sum of pixels for each cluster
    weighted_sum = flat_images.T @ resp

    # Normalize by the total weight assigned to each cluster
    cluster_pixel_probs = weighted_sum / (resp.sum(axis=0) + 1e-9)

    # Clip to avoid probabilities of exactly 0 or 1 (prevents log(0))
    cluster_pixel_probs = np.clip(cluster_pixel_probs, 1e-7, 1 - 1e-7)

    return cluster_pixel_probs.T # Shape (2, 784)

def compute_responsibilities(images, theta, pi):
    """
    E-step: Computes P(k | Xz) using the log-sum-exp trick for stability.
    """
    N = images.shape[0]
    flat_images = images.reshape(N, -1)

    # Log-likelihood of Bernoulli: x*log(p) + (1-x)*log(1-p)
    # log_p: (2, 784), flat_images: (N, 784)
    # result: (N, 2)
    log_theta = np.log(theta)
    log_one_minus_theta = np.log(1 - theta)

    # Vectorized computation of log-likelihood for all images
    log_lik = flat_images @ log_theta.T + (1 - flat_images) @ log_one_minus_theta.T

    # Add log of prior: log(P(X|k) * P(k)) = log P(X|k) + log P(k)
    weighted_log_lik = log_lik + np.log(pi + 1e-9)

    # Log-sum-exp trick for numerical stability when normalizing
    log_shift = np.max(weighted_log_lik, axis=1, keepdims=True)
    exp_weights = np.exp(weighted_log_lik - log_shift)
    resp = exp_weights / exp_weights.sum(axis=1, keepdims=True)

    return resp

def main():
    parser = argparse.ArgumentParser(description="EM for MNIST Clusters.")
    parser.add_argument("--first_cluster_digit", type=int, default=0)
    parser.add_argument("--second_cluster_digit", type=int, default=1)
    parser.add_argument("--n_iter", type=int, default=10)
    args = parser.parse_args()

    # Load data
    print("Fetching MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data, mnist.target.astype(int)

    # Filter for two specific digits
    mask = (y == args.first_cluster_digit) | (y == args.second_cluster_digit)
    images = X[mask]
    labels = y[mask]

    # Binarize and Normalize
    images = (images > 127.5).astype(float)
    target_labels = (labels == args.second_cluster_digit).astype(int)

    # Split
    train_images = images[:10000]
    test_images = images[10000:]
    test_labels = target_labels[10000:]

    # Initialize responsibilities randomly
    N_train = len(train_images)
    resp = np.random.dirichlet([10, 10], size=N_train)

    print(f"Training on {N_train} samples...")
    t1 = perf_counter()

    for _ in tqdm(range(args.n_iter)):
        # M-Step
        pi = get_apriori_prob(resp)
        theta = update_pixel_probs(train_images.reshape(-1, 28, 28), resp, pi)

        # E-Step
        resp = compute_responsibilities(train_images.reshape(-1, 28, 28), theta, pi)

    t2 = perf_counter()
    print(f"Training completed in {t2-t1:.4f} seconds")

    print(f"Testing on {len(test_images)} samples...")
    # Inference on Test Set
    test_resp = compute_responsibilities(test_images.reshape(-1, 28, 28), theta, pi)
    pred_labels = np.argmax(test_resp, axis=1)

    # EM is unsupervised; it might swap 0 and 1. Check both mappings.
    acc = (pred_labels == test_labels).mean()
    accuracy = max(acc, 1 - acc)

    print(f"Test set accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()