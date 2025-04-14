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
        margin_of_error (float): desired margin of error (positive float)
        confidence_level (float): desired confidence level (0-1)
        batch_size (int): number of samples to run in parallel batches
        verbose (bool): whether to print progress and results
    Returns:
        mean: mean of value returned by the process
    """
    MIN_SAMPLES = 100 # Minimum number of samples before assessing margin of error
    
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
            # Update mean, variance, delta
            update_variance(batch)
            variance = get_variance()
            if n >= MIN_SAMPLES:
                t_score = stats.t.ppf(1 - alpha / 2, n - 1)
                delta = t_score * np.sqrt(variance / n)
            if log_progress:
                elapsed_time = time.time() - start_time
                print(f'Delta: {delta:.5f}, Mean: {mean:.5f}, Time elaped: {seconds_to_hms(elapsed_time)}')
    runtime = time.time() - start_time
    if log_progress:
        print(f'\nProcess: {process.__name__}{args}')
        print(f'Time per process (s): {np.mean(batch_times) / batch_size}')
        print(f'Batch size: {batch_size}')
        print(f'Avg batch time: {seconds_to_hms(np.mean(batch_times))}')
        print('Total runtime (s):', runtime)
        print('Replication count: ', n)
        print('Variance: ', get_variance())
        print(f'Mean: {mean} +/- {rho} ({(1 - alpha) * 100}% Confidence)')
    return mean
