import time
import multiprocessing
from itertools import repeat

import numpy as np
from scipy import stats

def seconds_to_hms(seconds: float) -> str:
    """ Returns seconds in 'HH:MM:SS' format. """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def estimate_stochastic_mean(process, args=(), margin_of_error=0.1, confidence_level=0.95, batch_size=8, log_progress=True) -> float:
    """ Runs the given stochastic process in parallel batches until 
    a mean with the desired margin of error is found with the given 
    confidence level. Returns the mean.
    Args:
        process (Callable): function to run in parallel batches. The
            function must return a float value.
        args (Tuple): arguments to pass to the function
        margin_of_error (float): desired CI halfwidth (positive float)
        confidence_level (float): desired confidence level (0-1)
        batch_size (int): number of samples to run in parallel batches
        verbose (bool): whether to print progress and results
    Returns:
        mean: mean of value returned by the process
    """
    MIN_SAMPLES = 300 # Minimum number of samples before assessing margin of error
    
    rho = margin_of_error
    alpha = 1 - confidence_level
    n = 0
    mean = 0.0
    m2 = 0.0
    delta = float('inf') # Current margin of error
    batch_count = 1
    batch_times = []

    # Update variance using Welford's online algo
    def update_variance(x_batch):
        nonlocal n, mean, m2
        for xi in x_batch:
            n += 1
            delta_x = xi - mean
            mean += delta_x / n
            m2 += delta_x * (xi - mean)

    def get_variance():
        return m2 / (n - 1) if n >= 2 else float('inf')

    start_time = time.time()
    with multiprocessing.Pool() as pool:
        # Process batches until margin of error is small enough
        while delta > rho:
            # Process batch in parallel
            if log_progress:
                print(f'Processing batch {batch_count} (size {batch_size})...', end='\r')
            batch_start_time = time.time()
            batch = pool.starmap(process, repeat(args, batch_size))
            batch_count += 1
            batch_times.append(time.time() - batch_start_time)
            # Update mean, variance, error
            update_variance(batch)
            variance = get_variance()
            if n >= MIN_SAMPLES:
                t_score = stats.t.ppf(1 - alpha / 2, n - 1)
                delta = t_score * np.sqrt(variance / n)
            if log_progress:
                elapsed_time = time.time() - start_time
                print(f'Delta: {delta:.5f}, Mean: {mean:.5f}, Elapsed time: {seconds_to_hms(elapsed_time)}')
    runtime = time.time() - start_time
    if log_progress:
        print(f'\nProcess: {process.__name__}{args}')
        print(f'Time per process (s): {np.mean(batch_times) / batch_size}')
        print(f'Batch size: {batch_size}')
        print(f'Avg batch time: {seconds_to_hms(np.mean(batch_times))}')
        print(f'Total runtime: {seconds_to_hms(runtime)}')
        print(f'Replication count: {n}')
        print(f'Variance: {get_variance()}')
        print(f'Mean: {mean:.5f} +/- {rho} ({(1 - alpha) * 100}% Confidence)')
    return mean


def estimate_stochastic_stats(
    process, args=(), relative_margin_of_error=0.01,
    minimum_margin_of_error=None,
    confidence_level=0.95, batch_size=8, log_progress=True
) -> dict:
    """Estimate the mean of a stochastic dictionary-valued process until all
    keys are within the desired relative margin of error.

    Args:
        process (Callable): Function that returns a dict of float values.
        args (Tuple): Arguments to pass to the process.
        relative_margin_of_error (float): Relative margin of error for each key.
        confidence_level (float): Confidence level for the interval.
        batch_size (int): Number of processes per batch.
        log_progress (bool): Whether to log progress.

    Returns:
        dict: Dictionary of means for each key.
    """
    MIN_SAMPLES = 300
    alpha = 1 - confidence_level

    n = 0  # Sample count
    stats_dict = {}  # {key: {'mean': float, 'm2': float}}
    deltas = {}      # {key: current margin of error}

    batch_count = 1
    batch_times = []
    start_time = time.time()

    def update_variance(batch):
        nonlocal n
        for result in batch:
            n += 1
            for k, v in result.items():
                if k not in stats_dict:
                    stats_dict[k] = {'mean': 0.0, 'm2': 0.0}
                entry = stats_dict[k]
                delta = v - entry['mean']
                entry['mean'] += delta / n
                entry['m2'] += delta * (v - entry['mean'])

    def get_variance(k):
        entry = stats_dict[k]
        return entry['m2'] / (n - 1) if n >= 2 else float('inf')

    def all_deltas_within_bounds():
        # Only check deltas if there are enough samples
        if n < MIN_SAMPLES:
            return False
        # Check delta of each stat
        for k in stats_dict:
            delta_out_of_bounds = (
                # Delta is greater than given reltaive MoE
                deltas[k] > relative_margin_of_error * abs(stats_dict[k]['mean'])
                # And there is no minimum MoE or delta is greater than it
                and (minimum_margin_of_error is None or deltas[k] > minimum_margin_of_error)
            )
            if delta_out_of_bounds:
                return False
        # Every delta passed check
        return True

    with multiprocessing.Pool() as pool:
        while True:
            if log_progress:
                print(f'Processing batch {batch_count}...', end='\r')
            batch_start_time = time.time()
            batch = pool.starmap(process, repeat(args, batch_size))
            batch_times.append(time.time() - batch_start_time)
            update_variance(batch)

            # Update deltas
            deltas.clear()
            if n >= MIN_SAMPLES:
                t_score = stats.t.ppf(1 - alpha / 2, n - 1)
                for k in stats_dict:
                    variance = get_variance(k)
                    delta = t_score * np.sqrt(variance / n)
                    deltas[k] = delta
            else:
                for k in stats_dict:
                    deltas[k] = float('inf')

            if log_progress:
                elapsed = time.time() - start_time
                print(f'Batch {batch_count} complete. Elapsed: {seconds_to_hms(elapsed)}')
                for k in sorted(stats_dict):
                    mean_k = stats_dict[k]['mean']
                    delta_k = deltas[k]
                    print(f'\t{k}: {mean_k:.5f} ± {delta_k:.5f}')
                print()

            if all_deltas_within_bounds():
                break
            batch_count += 1

    means = {k: v['mean'] for k, v in stats_dict.items()}
    if log_progress:
        print(f'Final results after {batch_count} batches:')
        for k, mean in means.items():
            moe = deltas[k]
            print(f'  {k}: {mean:.5f} ± {moe:.5f}')
    return means
