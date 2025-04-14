import json
import os
import time
import multiprocessing
from itertools import repeat

import numpy as np
from scipy import stats

#------------------------- Description --------------------------#

# Generates a list of incentives for each bike count of each station. The
# first index is the station, and the second index is the bike count. The value at
# that index is the incentive for that station and bike count. 

# Incentives are generated based on the rental and return rates in the 
# bike system parameters file and saved to the incentives file.

# Negative incentive means the station needs rentals
# Positive incentive means the station needs returns

#-------------------------- Load Data --------------------------#

# Establish data directory
if os.path.exists('external'):
    BASE_PATH = 'external/'
else:
    BASE_PATH = ''

FAIL_COUNTS_FILEPATH = BASE_PATH + 'data/fail_counts.json'
INCENTIVES_FILEPATH = BASE_PATH + 'data/incentives.json'
BIKE_SYSTEM_PARAMS_FILEPATH = BASE_PATH + 'data/sim_params.json'

# Load bike system parameters for unpacking
with open(BIKE_SYSTEM_PARAMS_FILEPATH, 'r') as file:
    params = json.load(file)

# Number of stations
N = len(params['capacities'])
# Lists indexed by station (length N)
RENT_RATES = params['rent_rates']  # floats
RETURN_RATES = params['return_rates']  # floats
CAPACITIES = params['capacities']  # ints

#-------------------------- Constants --------------------------#

# Estimate fail counts (stochastic)
FAIL_COUNT_SIM_TIME_HORIZON = 2.0  # Hours
CONFIDENCE_LEVEL = 0.99  # (0-1)
MARGIN_OF_ERROR = 0.01  # (positive float)
MIN_SAMPLES = 10  # Minimum number of samples before checking margin of error
BATCH_SIZE = 2**14  # Number of samples to run in parallel (tune to CPU)

# Generate incentives (deterministic)
INCENTIVE_COST = 0.0  # Deducted from fail count reduction

#-------------------------- Formatting -------------------------#

def seconds_to_hms(seconds: float) -> str:
    """ Returns seconds in 'HH:MM:SS' format. """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

#-------------------------------- Estimate Fail Counts ------------------------------#

def estimate_stochastic_process_mean(process, args, margin_of_error=0.1, confidence_level=0.95, batch_size=2**12, verbose=True) -> float:
    """ Takes a stochastic process and runs it in parallel batches until 
    a mean with the desired margin of error is found with the given 
    confidence level. 
    Args:
        process (Callable): function to run in parallel batches
        args (Tuple): arguments to pass to the function
        margin_of_error (float): desired margin of error (positive float)
        confidence_level (float): desired confidence level (0-1)
        batch_size (int): number of samples to run in parallel batches
        verbose (bool): whether to print progress and results
    Returns:
        mean: mean of value returned by the process
    """
    rho = margin_of_error
    alpha = 1 - confidence_level
    n = 0
    mean = 0.0
    m2 = 0.0
    delta = float('inf') # Current margin of error

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
            batch = pool.starmap(process, repeat(args, batch_size))
            # Update mean, variance, delta
            update_variance(batch)
            variance = get_variance()
            if n >= MIN_SAMPLES:
                t_score = stats.t.ppf(1 - alpha / 2, n - 1)
                delta = t_score * np.sqrt(variance / n)
            if verbose:
                print(f'Delta: {delta:.5f}, Mean: {mean:.5f}')
                print(f'n: {n}, Variance: {variance:.5f}')
                print(f'Batch size: {len(batch)}, Time elapsed (s): {time.time() - start_time:.2f}')
    runtime = time.time() - start_time
    if verbose:
        print(f'\nProcess: {process.__name__}{args}')
        print('Total runtime (s):', runtime)
        print('Replication count: ', n)
        print('Variance: ', get_variance())
        print(f'Mean: {mean} +/- {rho} ({(1 - alpha) * 100}% Confidence)')
    return mean

def generate_fail_count(station: int, bike_count: int) -> float:
    """ Returns a sample of the number of failures at the given station in a 
    finite time horizon using a retrospective simulation. """
    # Get station parameters
    capacity = CAPACITIES[station]
    agg_rate = RETURN_RATES[station] + RENT_RATES[station]
    # Get probability that an arrival is a rental
    p_rent = RENT_RATES[station]/agg_rate
    # Set time and bike count
    T = FAIL_COUNT_SIM_TIME_HORIZON
    t = 0  # current time
    # Track time full/empty
    time_full, time_empty = 0, 0
    # Sub-simulation:
    while True:
        # Get holding time for next arrival
        tau = np.random.exponential(1/agg_rate)
        # Increment time, limited by horizon
        if t + tau <= T:
            t += tau
        else:
            tau = T - t
            t = T
        # Add holding time to full/empty times
        if bike_count == capacity:
            time_full += tau
        elif bike_count == 0:
            time_empty += tau
        # End simulation at time horizon
        if t == T:
            break
        # Determine arrival event and adjust bike count
        U = np.random.uniform(0, 1)
        if U < p_rent and bike_count > 0:
            bike_count -= 1
        elif bike_count < capacity:
            bike_count += 1
    # Estimate fail count conditioned on time full/empty
    fail_count = (
        time_empty * RENT_RATES[station]
        + time_full * RETURN_RATES[station]
    )
    # Get average fail count
    return fail_count

def estimate_all_fail_counts(verbose=True, save_to_file=True) -> list:
    if verbose:
        total_run_count = sum(CAPACITIES)
        run_count = 0
        start_time = time.time()
    fail_counts = []
    # Iterate through each station
    for station in range(N):
        # Generate fail counts for each bike count
        station_fail_counts = []
        for bike_count in range(CAPACITIES[station] + 1):
            # Generate incentive for each station and bike count
            fail_count = estimate_stochastic_process_mean(
                process=generate_fail_count, 
                args=(station, bike_count),
                margin_of_error=MARGIN_OF_ERROR,
                confidence_level=CONFIDENCE_LEVEL,
                batch_size=BATCH_SIZE,
                verbose=True
                )
            station_fail_counts.append(fail_count)
            if verbose:
                run_count += 1
                run_rate = run_count / (time.time() - start_time)
                eta = (total_run_count - run_count) / run_rate
                eta = seconds_to_hms(eta)
                print(f'fail_count[{station}][{bike_count}] ← {fail_count:.3f}\t ETA: {eta}', end='\r')
        # Add list of fail counts to main list
        fail_counts.append(station_fail_counts)
    if verbose:
        print('\nFail count estimation complete.')
        print('Total runtime:', seconds_to_hms(time.time() - start_time))
    if save_to_file:
        with open(FAIL_COUNTS_FILEPATH, 'w') as file:
            json.dump(fail_counts, file)
        print(f'Fail counts saved to {FAIL_COUNTS_FILEPATH}')
    return fail_counts

#-------------------------------- Calculate Incentives ------------------------------#

def calculate_incentive(station: int, bike_count: int, fail_counts: list) -> float:
    """ Calculates the incentive for a given station and bike count. """
    fail_count = fail_counts[station][bike_count] 
    
    # Determine performance by comparing fail counts
    if bike_count > 0:
        rent_fail_count = fail_counts[station][bike_count - 1]
        rent_fail_reduction = fail_count - rent_fail_count
        rent_performance = rent_fail_reduction - INCENTIVE_COST
    else:
        rent_performance = 0
    
    if bike_count < CAPACITIES[station]:
        return_fail_count = fail_counts[station][bike_count + 1]
        return_fail_reduction = fail_count - return_fail_count
        return_performance = return_fail_reduction - INCENTIVE_COST
    else:
        return_performance = 0
    
    # If both performances are positive, use higher one
    if rent_performance > 0 and return_performance > 0:
        if rent_performance > return_performance:
            return_performance = 0
        else:
            rent_performance = 0

    # Assign incentive based on performance
    if rent_performance > 0:
        return -1 * rent_performance
    if return_performance > 0:
        return return_performance
    return 0

def calculate_all_incentives(fail_counts: list, save_to_file=True) -> list:
    """ Calculates incentives for each station and bike count based on fail counts. """
    incentives = []
    # Iterate through each station
    for station in range(N):
        # Calculate incentives for each bike count
        station_incentives = []
        for bike_count in range(CAPACITIES[station] + 1):
            incentive = calculate_incentive(station, bike_count, fail_counts)
            station_incentives.append(incentive)
        # Add list of incentives to main list
        incentives.append(station_incentives)
    # Save to file
    if save_to_file:
        with open(INCENTIVES_FILEPATH, 'w') as file:
            json.dump(incentives, file)    
        print(f'Incentives saved to {INCENTIVES_FILEPATH}')    
    return incentives

#-------------------------------- Main -------------------------------#

def generate_incentives(import_fail_counts=False, save_incentives=True, verbose=True):
    if import_fail_counts:
        print('Importing fail counts...')
        with open(FAIL_COUNTS_FILEPATH, 'r') as file:
            fail_counts = json.load(file)
    else:
        print('Estimating fail counts...')
        fail_counts = estimate_all_fail_counts(verbose, save_to_file=True)
    
    print('Calculating incentives...')
    calculate_all_incentives(fail_counts, save_incentives)
    
def main():
    generate_incentives(
        import_fail_counts=False,
        save_incentives=True,
        verbose=True
    )

if __name__ == '__main__':
    main()