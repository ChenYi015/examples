import argparse
import ray
import random
import time

ray.init()


@ray.remote
def count_points_in_circle(n_samples):
    count = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            count += 1
    return count


def estimate_pi(n_samples, n_workers):
    samples_per_worker = n_samples // n_workers
    futures = [count_points_in_circle.remote(samples_per_worker)
               for _ in range(n_workers)]
    count = sum(ray.get(futures))
    return count / n_samples * 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=int,
        default=1000000000,
        metavar="M",
        help="Number of samples to estimate PI(default: 1000000000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        metavar="N",
        help="Number of workers(default: 10)"
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    n_samples = args.samples
    n_workers = args.workers
    print(f"Estimating PI with {n_samples} samples and {n_workers} workers.")
    start = time.time()
    pi = estimate_pi(n_samples, n_workers)
    end = time.time()
    print(f"Estimation of PI: {pi:.6f}.")
    print(f"Estimating time: {end - start:.2f}s.")
