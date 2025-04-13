import numpy as np
import json
import os
import time
from scipy import stats

# Generates a list of incentives for each bike count of each station. The
# first index is the station, and the second index is the bike count. The value at
# that index is the incentive for that station and bike count. 

# Incentives are generated based on the rental and return rates in the 
# bike system parameters file and saved to the incentives file.


#-------------------------- Load Data --------------------------#
# Establish data directory
if os.path.exists('external'):
    BASE_PATH = 'external/'
else:
    BASE_PATH = ''

INCENTIVES_FILEPATH = BASE_PATH + 'data/incentives.json'
BIKE_SYSTEM_PARAMS_FILEPATH = BASE_PATH + 'data/sim_params.json'

# Load bike system parameters for unpacking
with open(BIKE_SYSTEM_PARAMS_FILEPATH, 'r') as file:
    params = json.load(file)

# Number of stations
N = len(params['capacities'])
# Lists indexed by station (length N)
RENT_RATES = params['rent_rates']
RETURN_RATES = params['return_rates']  # floats
CAPACITIES = params['capacities']  # ints

#-------------------------- Constants --------------------------#
# Estimate fail counts (stochastic)
FAIL_COUNT_SIM_TIME_HORIZON = 2.0  # Hours
CONFIDENCE_LEVEL = 0.95  # (0-1)
MARGIN_OF_ERROR = 0.01  # (positive float)
MIN_SAMPLES = 10  # Minimum number of samples before checking margin of error
# Generate incentives (deterministic)
INCENTIVE_COST = 0.5  # Deducted from fail count reduction

def generate_single_incentive(station: int, bike_count: int) -> float:
    # Estimate fail counts for adding/removing a bike
    fail_count = estimate_fail_count(station, bike_count, bike_offset=0)
    rent_fail_count = estimate_fail_count(station, bike_count, bike_offset=-1)
    return_fail_count = estimate_fail_count(station, bike_count, bike_offset=1)
    # Calculate failure reductions
    rent_fail_reduction = fail_count - rent_fail_count
    return_fail_reduction = fail_count - return_fail_count
    # Subtract incentive cost to determine performance
    rent_performance = rent_fail_reduction - INCENTIVE_COST
    return_performance = return_fail_reduction - INCENTIVE_COST
    # Determine station's incentive based on performance
    # ToDo: Test for cases where both reductions exceed 0
    # Incentive is negative if rentals are incentivized
    if rent_performance > 0:
        return -1 * rent_performance
    # Incentive is positive if returns are incentivized
    elif return_performance > 0:
        return return_performance
    # Zero indicates no incentive
    else: 
        return 0

def estimate_fail_count(station: int, bike_count: int) -> float:
    """ Estimates the number of failures at the given station in a finite time horizon
    using a retrospective simulation. """
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



def get_confidence_interval(process, kwargs : dict):
    # Using the Fixed-Width Confidence Interval, replicate simulations,
    # until the margin of error delta is less than rho
    # confidence level is 1 - alpha
    
    rho = MARGIN_OF_ERROR
    alpha = 1 - CONFIDENCE_LEVEL
    n = 0 # sample count
    mean = 0.0
    m2 = 0.0  # The sum of squares of differences from the mean
    delta = float('inf')  # margin of error
    runtimes = []

    def update_variance(x):
        nonlocal n, mean, m2
        n += 1
        delta_x = x - mean
        mean += delta_x / n
        m2 += delta_x * (x - mean)

    def get_variance():
        if n < 2:
            return float('inf')  # Not enough samples to compute variance
        return m2 / (n - 1)

    # Start sampling
    total_start_time = time.time()
    while delta > rho:
        start_time = time.time()
        sample = process(**kwargs)
        runtimes.append(time.time() - start_time)
        update_variance(sample)
        variance = get_variance()
        if n >= MIN_SAMPLES:  # Calculate t-score and margin of error only with sufficient samples
            t_score = stats.t.ppf(1 - alpha / 2, n - 1)
            delta = t_score * np.sqrt(variance / n)
        print(f'Delta: {delta:.5f}, Mean: {mean:.5f}')
    total_runtime = time.time() - total_start_time
    print(f'\nProcess: {process.__name__}')
    print(f'{kwargs.items()}')
    print('Total runtime (s):', total_runtime)
    print('Average replication runtime (s): ', sum(runtimes)/len(runtimes))
    print('Replication count: ', n)
    print('Variance: ', get_variance())
    print(f'Mean: {mean} +/- {rho} ({(1 - alpha) * 100}% Confidence)')
    return mean


def main():
    get_confidence_interval(estimate_fail_count, {
        'station': 0,
        'bike_count': 1
    })


if __name__ == '__main__':
    main()