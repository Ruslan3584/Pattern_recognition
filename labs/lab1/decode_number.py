import argparse
from time import perf_counter

import cv2
import numpy as np


def generate_number_img(number: int, height: int, width: int) -> np.ndarray:
    """Convert a number to a binary (0/1) image."""
    init_height = 25
    init_width = 25
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


def main():
    """Decode a number image with Bernoulli noise."""
    parser = argparse.ArgumentParser(description="Decode noisy image of a number")
    parser.add_argument(
        "--number",
        type=int,
        required=True,
        help="Number to encode and decode",
        choices=list(range(10)),
    )
    parser.add_argument(
        "--number_height",
        type=int,
        default=50,
        required=False,
        help="Height of the number image.",
    )
    parser.add_argument(
        "--number_width",
        type=int,
        default=100,
        required=False,
        help="Width of the number image.",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        required=True,
        help="Probability of Bernoulli noise. (0.0 to 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
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

    t1 = perf_counter()

    # Set a random seed
    np.random.seed(args.seed)

    # Prepare standards of numbers
    standards = np.zeros((10, args.number_height, args.number_width), dtype=np.uint8)
    for i in range(10):
        standards[i] = generate_number_img(i, args.number_height, args.number_width)

    number_img = standards[args.number]

    # Apply binomial noise to the number image
    noise = np.random.binomial(n=1, p=args.noise_level, size=number_img.shape).astype(
        np.uint8
    )
    number_img_noised = cv2.bitwise_xor(number_img, noise)

    # Decode the number
    xored_standards_w_number = np.bitwise_xor(standards, number_img_noised).astype(
        np.float32
    )
    xored_standards_w_number *= args.noise_level
    xored_standards_w_number[xored_standards_w_number == 0] = 1 - args.noise_level

    decoded_number = np.argmax(np.sum(np.log10(xored_standards_w_number), axis=(1, 2)))

    t2 = perf_counter()

    print(f"Time: {t2 - t1} sec")

    print(f"Decoded number: {decoded_number}")

    # Save results
    cv2.imwrite("number_img.png", (number_img * 255).astype(np.uint8))
    cv2.imwrite(
        f"number_img_noised_{args.noise_level}.png",
        (number_img_noised * 255).astype(np.uint8),
    )


if __name__ == "__main__":
    main()
