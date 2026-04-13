import argparse

import cv2
import numpy as np
from time import perf_counter
from tqdm import tqdm


def generate_number_img(number: int, height: int, width: int) -> np.ndarray:
    """Convert a number to a binary (0/1) image."""
    init_height = 20
    init_width = 20
    img = np.zeros((init_height, init_width), dtype=np.uint8)

    text = str(number)
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = min(init_height, init_width) / 25
    thickness = max(1, int(font_scale * 2))

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (init_width - text_w) // 2
    y = (init_height + text_h) // 2

    cv2.putText(img, text, (x, y), font, font_scale, 1, thickness, cv2.LINE_AA)

    # Resize
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    # Ensure strict binary (0/1)
    img = (img > 0).astype(np.uint8)

    return img


def generate_random_digits_seq(
    n_digits: int, histogram: np.ndarray, standards: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random sequence of digits."""
    digits = np.random.choice(10, size=n_digits, p=histogram)
    digits_img = np.hstack(standards[digits])
    return digits, digits_img


def check_division_strategy(a: np.ndarray, b: int = 3) -> bool:
    """Check if the sum of elements at every b-th index is >= the rest."""
    sum_k2 = sum(a[::b])
    total_sum = sum(a)

    return 2 * sum_k2 >= total_sum


def calculate_probability(
    digits_splited_img: np.ndarray,
    n_digits: int,
    noise_level: float,
    histogram: np.ndarray,
    standards: np.ndarray,
) -> np.ndarray:
    if n_digits <= 0:
        return np.array([])

    # 1. Get the probability distribution for the first digit
    res = recognize(digits_splited_img[0], noise_level, histogram, standards)

    if n_digits == 1:
        return res

    # 2. Iteratively compute the distribution of the sum using Discrete Convolution
    for i in range(1, n_digits):
        z = recognize(digits_splited_img[i], noise_level, histogram, standards)
        res = np.convolve(res, z)

    return res


def recognize(
    digits_splited_img: np.ndarray,
    noise_level: float,
    histogram: np.ndarray,
    standards: np.ndarray,
) -> np.ndarray:
    """Recompute probabilities."""
    probabs = np.empty(10)
    d = standards ^ digits_splited_img
    for ks in range(10):
        stepin = d - (standards[ks] ^ digits_splited_img)
        if np.sum(stepin[ks]) == 0:
            h = np.zeros(10)
            h[ks] = 1
            return h
        sum_ki = histogram * (noise_level / (1 - noise_level)) ** np.sum(stepin, axis=0)
        probabs[ks] = histogram[ks] / np.sum(sum_ki)
    return probabs


def main():
    parser = argparse.ArgumentParser(description="Generate a number image.")
    parser.add_argument(
        "--n_digits", type=int, required=True, help="Number of digits to generate."
    )
    parser.add_argument(
        "--divisor", type=int, required=False, default=3, help="Check division by."
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        required=True,
        help="Bernoulli noise level. (0 to 1).",
    )
    parser.add_argument(
        "--height_digit",
        type=float,
        required=False,
        help="Height of the digit image.",
        default=20,
    )
    parser.add_argument(
        "--width_digit",
        type=float,
        required=False,
        help="Width of the digit image.",
        default=20,
    )
    parser.add_argument(
        "--n_iter", type=int, required=False, help="Test multiple times.", default=1
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=100,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.noise_level > 1 or args.noise_level < 0:
        raise Exception("noise_level must be in range (0,1)")

    # Avoid division by zero
    if args.noise_level == 0:
        args.noise_level = 0.00001
    if args.noise_level == 1:
        args.noise_level = 0.99999

    n_success_divisions = 0

    np.random.seed(args.seed)

    debug_inputs = []
    debug_inputs_str = []
    debug_inputs_noised = []
    debug_div_result = []

    t1 = perf_counter()

    for _ in tqdm(range(args.n_iter)):
        # Prepare standards of numbers
        standards = np.zeros((10, args.height_digit, args.width_digit), dtype=np.uint8)
        for i in range(10):
            standards[i] = generate_number_img(i, args.height_digit, args.height_digit)

        # Generate histogram (digits probabilities)
        histogram = np.random.randint(0, 100, 10)

        # Normalize histogram
        histogram = histogram / np.sum(histogram)
        digits, digits_img = generate_random_digits_seq(
            args.n_digits, histogram, standards
        )
        digits_sum = np.sum(digits)

        # Generate noise
        noise = np.random.binomial(
            size=args.height_digit * args.width_digit * args.n_digits,
            n=1,
            p=args.noise_level,
        ).reshape((args.height_digit, args.width_digit * args.n_digits))

        # Apply noise to the digits image
        digits_img_noised = np.bitwise_xor(digits_img, noise)

        # Split noised digits into separate images
        digits_splited_img = np.hsplit(digits_img_noised, args.n_digits)

        # Calculate probabilities
        probs = calculate_probability(
            digits_splited_img, args.n_digits, args.noise_level, histogram, standards
        )

        number_str = "".join(map(str, digits))

        # Check division
        divided = False
        if digits_sum != 0:
            if (
                check_division_strategy(probs, args.divisor) == True
                and digits_sum % args.divisor == 0
            ) or (
                check_division_strategy(probs, args.divisor) == False
                and digits_sum % args.divisor != 0
            ):
                n_success_divisions += 1
                divided = True
            else:
                divided = False

        debug_inputs_str.append(number_str)
        debug_inputs.append(digits_img)
        debug_inputs_noised.append(digits_img_noised)
        debug_div_result.append(divided)

    for i in range(args.n_iter):
        print(
            "Number: ",
            debug_inputs_str[i],
            f" Divided by {args.divisor}: ",
            debug_div_result[i],
        )

    # Save images
    for i in range(args.n_iter):
        cv2.imwrite(
            f"digits_img__iter_{i}.png", (debug_inputs[i] * 255).astype(np.uint8)
        )
        cv2.imwrite(
            f"digits_img_noised__iter_{i}_div_{debug_div_result[i]}.png",
            (debug_inputs_noised[i] * 255).astype(np.uint8),
        )

    print("Success divisions: ", f"{n_success_divisions}/{args.n_iter}")

    t2 = perf_counter()
    print("time : ", t2 - t1, "sec")


if __name__ == "__main__":
    main()
