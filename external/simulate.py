# Main simulation
import json
from dataclasses import dataclass, field
import logging
import time
import os
from queue import PriorityQueue
import bisect

import numpy as np

from tools import estimate_stochastic_mean, estimate_stochastic_stats
        
# Unless otherwise indicated:
# - "station" refers to station index
# - "time" refers to simulated time measured in hours

# Establish data directory for dev builds
if os.path.exists('external'):
    BASE_PATH = 'external/'
else:
    BASE_PATH = ''

#=============================== STATIC PARAMETERS ===========================================

#-------------------- CONFIGURATION --------------------
START_TIME = 16.0  # Time of day (HH) when the simulation begins                                                     
                                        
AGENT_SEARCH_BRANCH_FACTOR = 4  # The number of nearest stations (with rewards if biking) to
                                # search when expanding a node
AGENT_MAX_SEARCH_DEPTH = 4  # The max depth of the agent's search tree
AGENT_WAIT_TIME = 5.001/60  # The length of time in hours the agent waits when no station found
AGENT_WAIT_TIME_LENIENCY = 2/60 # If the wait time ends at most this much time before
                                # the update, then agent extends wait till update
AGENT_MAX_WALK_TIME = 12/60  # The maximum time in hours the agent will walk to a station
INCENTIVE_COST = 0.1    # The number of failures that must be mitigated to warrant
                        # incentivizing a station
INCENTIVE_PRECISION = 2     # The number of decimal places to round incentives to
INCENTIVE_UPDATE_INTERVAL = 0.25    # The period of time in hours between incentives updates

#----------------------- TESTING -----------------------
HARD_STOP_TIME = 0  # Stop simulation after this many hours. If 0, don't stop until finished.

#------------------- SPECIAL VALUES --------------------
NULL_STATION = -1  # Represents null station index
END_TRIP = -2  # Action that indicates there is no more time for trips
LARGE_INT = 1_000_000  # Used like inf to indicate bad fail count for impossible bike counts
HIT = '\u2b55 ' 
MISS = '\U000026AB ' 
WARNING = '\U0001F535 '
ERROR = '\U0001F534 '

#---------------- LOAD USER PARAMETERS -----------------
USER_PARAMS_FILEPATH = BASE_PATH + 'data/user_params.json'
with open(USER_PARAMS_FILEPATH, 'r') as file:
    USER_PARAMS = json.load(file)
    
#----------- Agent Parameters -----------
START_STATION = USER_PARAMS['start_station']  # Agent start station index
FINAL_STATION = USER_PARAMS['end_station']  # Agent final station index
EXCURSION_TIME = USER_PARAMS['excursion_time']  # Length of excursion in hours
AGENT_INTELLIGENCE = USER_PARAMS['agent_mode']  # 'basic', 'smart'
#------------ Sim Parameters ------------
EMPTY_BIAS = USER_PARAMS['empty_bias']  # Bias towards emptying stations (0-1)
FULL_BIAS = USER_PARAMS['full_bias']  # Bias towards filling stations (0-1)
WARM_UP_TIME = USER_PARAMS['warmup_time']  # The number of hours that the simulation runs before starting the agent
SIM_MODE = USER_PARAMS['sim_mode']  # 'single_run' or 'batch'
# Single Run
USE_STATIC_SEED = USER_PARAMS['use_static_seed']
SEED = USER_PARAMS['seed']  # Unsigned 32 bit int
# Batch
CONFIDENCE_LEVEL = USER_PARAMS['confidence_level']
PARALLEL_BATCH_SIZE = USER_PARAMS['parallel_batch_size']
BATCH_MODE = USER_PARAMS['batch_mode']  # 'precision_based' or 'fixed_sample_size'
# - Fixed sample size
BATCH_SIZE = USER_PARAMS['batch_size']  # Number of simulation replications
# - Precision based
MIN_SAMPLE_SIZE = USER_PARAMS['min_sample_size']
RELATIVE_MARGIN_OF_ERROR = USER_PARAMS['relative_margin_of_error']
ABSOLUTE_MARGIN_OF_ERROR = USER_PARAMS['absolute_margin_of_error']
MAX_RUNTIME = USER_PARAMS['max_runtime']


#--------------- BIKE SYSTEM PARAMETERS ----------------

BIKE_SYSTEM_PARAMS_FILEPATH = BASE_PATH + 'data/sim_params.json'

# Load bike system parameters for unpacking
with open(BIKE_SYSTEM_PARAMS_FILEPATH, 'r') as file:
    sim_params = json.load(file)

# Number of stations
N = len(sim_params['capacities'])

# Lists indexed by station (length N)
#RENT_RATES = [x * 10 for x in sim_params['rent_rates'] ] # floats
RENT_RATES = sim_params['rent_rates']
RETURN_RATES = sim_params['return_rates']  # floats
INITIAL_BIKE_COUNTS = sim_params['initial_bike_counts']  # ints
CAPACITIES = sim_params['capacities']  # ints

# 2D lists indexed by station ([N x N] float matrix)
DEST_PROBS = sim_params['dest_probs']  # Destination probability matrix
BIKE_TIMES = sim_params['bike_times']
WALK_TIMES = sim_params['walk_times']

def get_near_stations_list(travel_time_matrix: list) -> list:
    """ Takes a travel time matrix and returns a 2D list where each row corresponds
    to a start station, and contains indices of every station sorted by distance to 
    the start station. """
    nearest_matrix = []
    for i, times in enumerate(travel_time_matrix):
        # Create a list of (index, time) pairs for this row of times
        indexed_times = [(j, t) for j, t in enumerate(times)]
        # Sort by travel time
        indexed_times = sorted(indexed_times, key=lambda x: x[1])
        # Get station indices
        sorted_indices = [index for index, _ in indexed_times]
        # Add row to new matrix, excluding first station (itself)
        nearest_matrix.append(sorted_indices[1:])
    return nearest_matrix

# List of nearest stations (excluding itself, so [N x (N-1)] int matrix)
NEAR_BIKE_STATIONS = get_near_stations_list(sim_params['bike_times'])
NEAR_WALK_STATIONS = get_near_stations_list(sim_params['walk_times'])

# Load fail counts from file
FAIL_COUNTS_FILEPATH = BASE_PATH + 'data/fail_counts.json'
with open(FAIL_COUNTS_FILEPATH, 'r') as file:
    # FAIL_COUNTS[<station>][<bike_count>]
    FAIL_COUNTS = json.load(file)

# Load incentives from file
INCENTIVES_FILEPATH = BASE_PATH + 'data/incentives.json'
with open(INCENTIVES_FILEPATH, 'r') as file:
    # INCENTIVES[<station>][<bike_count>]
    INCENTIVES = json.load(file)

def get_incentives_with_cost():
    """ Returns incentives with incentive cost deducted. Incentive cost
    is deducted from absolute value of incentive and does not go beyond
    zero. """
    new_incentives = []
    for station_incentives in INCENTIVES:
        new_station_incentives = []
        for incentive in station_incentives:
            # Subtract cost if positive
            if incentive > 0:
                incentive = max(0, incentive - INCENTIVE_COST)
            # Add cost if negative
            elif incentive < 0:
                incentive = min(0, incentive + INCENTIVE_COST)
            # Round incentive
            incentive = round(incentive, INCENTIVE_PRECISION)
            # Add to list
            new_station_incentives.append(incentive)
        new_incentives.append(new_station_incentives)
    return new_incentives
    
# Subtract cost from incentives 
INCENTIVES = get_incentives_with_cost()

# Load bike distances
BIKE_DISTANCES_FILEPATH = BASE_PATH + 'data/bike_distances.json'
with open(BIKE_DISTANCES_FILEPATH, 'r') as file:
    BIKE_DISTANCES = json.load(file)

#============================================================================================

#----------------------------------------- LOGGING ------------------------------------------
# Levels: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
SINGLE_RUN_LOG_LEVEL = logging.INFO
BATCH_LOG_LEVEL = logging.ERROR
PRINT_BATCH_PROGRESS = True
# Time range for debug log (HH)
DEBUG_START_TIME = 17 + 20/60
DEBUG_END_TIME = 17 + 30/60

WRITE_LOG_FILE = True

def generate_log_filepath(seed):
    """ Generates a log filepath in the format data/YYMMDD_HHMM_s<seed>.log """
    log_dir = os.path.join(BASE_PATH[:-1], 'logs')
    
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp
    timestamp = time.strftime('%y%m%d_%H%M')
    if seed is None:
        seed = 'batch'
    else:
        seed = f's{seed}'
    base_name = f'{timestamp}_{seed}'
    ext = '.log'
    filename = base_name + ext
    counter = 2

    # Increment filename if it already exists
    while os.path.exists(os.path.join(log_dir, filename)):
        if '-' in base_name:
            base_name = base_name.rsplit('-', 1)[0]  # Remove last -n if exists
        filename = f"{base_name}-{counter}{ext}"
        counter += 1

    return os.path.join(log_dir, filename)

LOG_FILEPATH = generate_log_filepath(SEED)
print(LOG_FILEPATH, end='')

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(SINGLE_RUN_LOG_LEVEL) 
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
if WRITE_LOG_FILE:
    file_handler = logging.FileHandler(LOG_FILEPATH, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

### logging.disable()

#============================================================================================

@dataclass(frozen=True)
class Node:
    """ A search node for the bike rebalancing agent """
    action: int        # The index of the first station (after the root) in this node's branch
    station: int       # Current station index
    mode: str          # Upcoming action: ('walk', 'bike')
    root_mode: str     # Mode of root node
    time: float        # Current time
    reward: float      # Total reward
    depth: int=0       # Depth in search tree
    prev: 'Node'=None  # Previous node
    bike_differences: dict=field(default_factory=dict)  # {<station>: <bike_difference>} Track net bikes
                                                        # the agent has added/removed from each station
    
    
class Agent:
    """ The bike rebalancing agent. The agent iteratively takes trips between stations. """
    def __init__(self):
        """ An agent iteratively takes trips between stations, alternating between biking and walking. """
        self.station = START_STATION  # Current station index
        self.final_station = FINAL_STATION  # The station the agent wants to end at when time is done
        self.mode = 'bike'
        self.end_time = START_TIME + EXCURSION_TIME  # Agent should arrive at final station by this time
        self.reward = 0  # Total rewards earned
        self.can_rent_bike = True # True if the agent cant rent a bike from the current station
                                  # (becomes False after delivering a bike)

    def update(self, new_station: int, reward_gain: float) -> None:
        """ Updates the agent's state based on the given trip data. """
        # Update station
        self.station = new_station
        # Alternate mode and set rent restriction
        if self.mode == 'walk':
            self.mode = 'bike'
            self.can_rent_bike = True
        elif self.mode == 'bike':
            self.mode = 'walk'
            self.can_rent_bike = False
        # Adjust reward
        self.reward += reward_gain    

    def get_action(self, bike_counts: list, incentives: list, current_time: float) -> int:
        """ Returns the next station the agent will travel to. """
        # Use smart action if agent is smart, otherwise use basic action
        if AGENT_INTELLIGENCE == 'smart':
            return self._get_smart_action(bike_counts, incentives, current_time)
        elif AGENT_INTELLIGENCE == 'basic':
            return self._get_basic_action(incentives, current_time)

    def _get_basic_action(self, incentives: list, current_time: float) -> int:
        """" Returns the next station the agent will travel without prediction. """
        # Set nearest stations and trip times based on mode
        if self.mode == 'bike':
            near_stations = NEAR_BIKE_STATIONS
            trip_times = BIKE_TIMES
        elif self.mode == 'walk':
            near_stations = NEAR_WALK_STATIONS
            trip_times = WALK_TIMES
            # Track closest walk station without return incentive
            nearest_walk_station = None
        # Queue stations prioritized by incentive per time
        station_queue = PriorityQueue()
        # Track whether there's time to reach final station
        has_time_to_finish_excursion = False
        for end_station in near_stations[self.station]:
            # Ensure there is enough time to reach final station
            return_time = (
                current_time 
                + trip_times[self.station][end_station] 
                + WALK_TIMES[end_station][self.final_station]
            )
            if return_time > self.end_time:
                continue
            else:
                has_time_to_finish_excursion = True
            # Ensure not walking to far away station
            if self.mode == 'walk' and trip_times[self.station][end_station] > AGENT_MAX_WALK_TIME:
                break  # Iterating through nearest stations, so trips only get longer
            # Ensure there is no counter-incentive
            incentive = incentives[end_station]
            if self.mode == 'walk':
                incentive *= -1  # Invert return incentive for walking
            # If walking, set nearest walk station
            if incentive >= 0 and self.mode == 'walk' and nearest_walk_station is None:
                nearest_walk_station = end_station
            if incentive <= 0:
                continue
            # Add station to queue
            value = incentive / trip_times[self.station][end_station]
            station_queue.put((-1 * value, end_station))
        # If there's no time to finish the excursion, end trip
        if not has_time_to_finish_excursion:
            return END_TRIP
        # No stations found
        if station_queue.empty():
            # No valid incentivized walk stations
            if self.mode == 'walk':
                # If the agent can rent a bike, just wait 
                if self.can_rent_bike:
                    return NULL_STATION
                # Otherwise, check for nearest walk station  
                if nearest_walk_station is not None:
                    return nearest_walk_station
                # No valid walk station
                else:
                    # If no time to wait, end trip
                    logger.warning(f'\t{WARNING} No valid walk station')
                    return_time = (
                        current_time 
                        + AGENT_WAIT_TIME
                        + WALK_TIMES[self.station][self.final_station]
                    )
                    if return_time > self.end_time:
                        return END_TRIP
                    # Otherwise wait
                    else:
                        return NULL_STATION
            # If no stations to bike to, wait
            elif self.mode == 'bike':
                return NULL_STATION
        # Return station with highest incentive per time 
        action = station_queue.get()[1]
        return action 

    def _get_smart_action(self, bike_counts: list, incentives: list, current_time: float) -> int:
        """ Returns the next station the agent will travel to using a search tree. """
        #---------------------------- Estimate future incentives ---------------------------
        # [(start_time, incentives0), (update1_time, incentives1), ...]
        predicted_incentives = [(current_time, incentives.copy())]
        prev_time = current_time
        prev_bike_counts = bike_counts
        # Get first future incentive update time
        new_time = np.ceil(current_time / INCENTIVE_UPDATE_INTERVAL) * INCENTIVE_UPDATE_INTERVAL
        # Iterate through update times and predict incentives
        while (new_time <= self.end_time):
            # Get bike counts and incentives for new time
            new_bike_counts = []
            new_incentives = []
            for station in range(N):
                new_bike_count = estimate_bike_count(station, prev_bike_counts[station], new_time - prev_time)
                new_incentive = INCENTIVES[station][new_bike_count]
                new_bike_counts.append(new_bike_count)
                new_incentives.append(new_incentive)
            # Add incentives for new time to cache
            predicted_incentives.append( (new_time, new_incentives) )
            # Save bike counts and update time to next incentives update
            prev_bike_counts = new_bike_counts
            prev_time = new_time
            new_time += INCENTIVE_UPDATE_INTERVAL
        #------------------------- Set up tree search ------------------------------
        # Use current time for root node
        root_time = current_time
        # Track best node
        best_node = None
        
        def is_best(node: Node):
            """ Returns True if node is better than best_node"""
            if best_node == 0:
                logger.error(f'{ERROR} Attempted to evaluate best_node None')
                return True
            # Get elapsed times
            best_node_time = best_node.time - root_time
            node_time = node.time - root_time
            # Calculate reward rates
            best_reward_rate = best_node.reward / best_node_time if best_node_time > 0 else 0
            new_reward_rate = node.reward / node_time if node_time > 0 else 0
            # Compare reward rates
            if new_reward_rate > best_reward_rate:
                return True

        #------------------------------ Tree Search -------------------------------
        # Create root search node
        root_node = Node(NULL_STATION, self.station, self.mode, self.mode, root_time, reward=0)
        best_node = root_node
        search_queue = [root_node]
        # Search until queue is empty
        while search_queue:
            node = search_queue.pop()  # DFS
            #---------------- Max depth reached ---------------
            if node.depth == AGENT_MAX_SEARCH_DEPTH:
                # Evaluate current node
                if is_best(node):
                    best_node = node
            #----------------- Expand node --------------------
            else:
                new_nodes = self._expand_node(node, bike_counts, root_time, predicted_incentives)
                search_queue += new_nodes
                # Log tree expansion
                if logger.level <= logging.DEBUG and (DEBUG_START_TIME <= root_time <= DEBUG_END_TIME):
                    logger.debug(f'{node.station}, time: {format_time(node.time)}, reward: {node.reward}, {node.mode} to {[n.station for n in new_nodes]}')
                #--------------------- No trips available  --------------------
                if not new_nodes:
                    #------------- Evaluate current node --------------
                    # Ensure there is a reward in this branch
                    # ToDo: is this check necessary now?
                    if node.reward > 0:
                        # Evaluate current node
                        if is_best(node):
                            best_node = node
                    #---------- No bike trips so far in branch ----------
                    else:  # reward == 0
                        # If there are more nodes to expand, continue search
                        if search_queue:
                            continue
                        # Otherwise check for best node
                        elif best_node.action != NULL_STATION:
                            return best_node.action
                        #----------------- No best node or trips found ------------
                        # Estimate return time after incentive update
                        update_time = np.ceil(root_time / INCENTIVE_UPDATE_INTERVAL) * INCENTIVE_UPDATE_INTERVAL
                        return_time_after_update = update_time + WALK_TIMES[self.station][self.final_station]
                        #------ Wait for update if there's time -----
                        if self.end_time >= return_time_after_update:
                            return NULL_STATION   
                        #------------ Otherwise end trip ------------
                        else:
                            logger.debug(f'eta: {format_time(return_time_after_update)}, root time: {format_time(root_time)}')
                            return END_TRIP
        #--------------------- Best node found ------------------------
        if best_node.action != NULL_STATION:
            # Return best action
            return best_node.action
        #--------------- No best node, end excursion ------------------
        # If no best node was found, then agent doesn't have time left
        else:
            return END_TRIP  # Finish excursion       

    def _expand_node(self, node: Node, root_bike_counts: list, root_time: float, predicted_incentives: list) -> list:
        # Get incentive times
        # ToDo: change predicted_incentives to tuple of lists to avoid this step
        incentive_times = [time_incentives_pair[0] for time_incentives_pair in predicted_incentives]
        new_nodes = []
        #------------------------------------- Walk -------------------------------------------
        if node.mode == 'walk':
            #-------------------------- Get incentives -------------------------
            # For walking, get incentives using predicted end trip time
            incentives = []
            for end_station in range(N):
                # Get time of incentive update before trip end, along with predicted incentives for that time
                end_trip_time = node.time + WALK_TIMES[node.station][end_station]
                predicted_incentives_index = bisect.bisect_right(incentive_times, end_trip_time) - 1
                update_time, incentives_for_time = predicted_incentives[predicted_incentives_index]
                # Estimate bike count using bike differences if station visited previously
                if end_station in node.bike_differences:
                    bike_count = estimate_bike_count(end_station, root_bike_counts[end_station], update_time - root_time)
                    bike_count += node.bike_differences[end_station]
                    incentives.append(INCENTIVES[end_station][bike_count])
                # Otherwise use precalculated incentive
                else:
                    incentives.append(incentives_for_time[end_station])
            #------------------------ Adjust walk time -------------------------
            # If no returns incentivized, extend max walk time to next update
            # (If there are returns incentivized, it's not worth walking for too long)
            max_walk_time = AGENT_MAX_WALK_TIME
            if max(incentives) <= 0:  # no return incentivized
                next_update_time = np.ceil(node.time / INCENTIVE_UPDATE_INTERVAL) * INCENTIVE_UPDATE_INTERVAL
                time_until_update = next_update_time - node.time
                max_walk_time = max(time_until_update, max_walk_time)
            #----------------------- Assess Stations ---------------------------
            # Get reward rates for incentivized stations and travel times for neutral stations
            station_reward_rate_pairs = []  # Incentivized stations
            station_travel_time_pairs = []  # Neutral stations
            for end_station, incentive in enumerate(incentives):
                # Validate station
                walk_time = WALK_TIMES[node.station][end_station]
                return_time = node.time + walk_time + WALK_TIMES[end_station][self.final_station]
                if (
                    # Trip exceeds max walk time
                    walk_time > max_walk_time
                    # Return time exceeds excursion end time
                    or return_time > self.end_time
                    # Station is itself
                    or node.station == end_station
                ):
                    # Skip station
                    continue
                # Get reward rate for every incentivized station
                if incentive < 0:
                    reward = get_reward(incentive, 'rent')
                    reward_rate = reward / WALK_TIMES[node.station][end_station]
                    station_reward_rate_pairs.append( (end_station, reward_rate) )
                # Get travel times for every neutral station
                elif incentive == 0:
                    station_travel_time_pairs.append( (end_station, WALK_TIMES[node.station][end_station]) )    
            #---------------------- Get best stations --------------------------
            WALK_BRANCH_FACTOR = AGENT_SEARCH_BRANCH_FACTOR
            new_stations = []
            # Get incentivized stations to expand to
            if station_reward_rate_pairs:
                # Sort stations by descending reward rate
                station_reward_rate_pairs = sorted(station_reward_rate_pairs, key=lambda x: x[1], reverse=True)
                stations = [pair[0] for pair in station_reward_rate_pairs]
                # Add stations up to branch factor
                new_stations.extend(stations[:WALK_BRANCH_FACTOR])
            # Not enough incentivized stations, try neutral
            if len(new_stations) < WALK_BRANCH_FACTOR:
                if station_travel_time_pairs:
                    # Sort stations by ascending travel time
                    station_travel_time_pairs = sorted(station_travel_time_pairs, key=lambda x: x[1])
                    # Validate and add station to new stations
                    for end_station, _ in station_travel_time_pairs:
                        trip_end_time = node.time + WALK_TIMES[node.station][end_station]
                        end_bike_count = estimate_bike_count(end_station, root_bike_counts[end_station], trip_end_time - root_time)
                        if self._validate_station(end_station, end_bike_count, node.mode):
                            new_stations.append(end_station)
                        # Add stations until branching factor is met
                        if len(new_stations) == WALK_BRANCH_FACTOR:
                            break
            #-------------------- Instantiate new nodes ----------------------
            for end_station in new_stations:
                # Adjust bike difference for end station
                bike_differences = node.bike_differences.copy()
                if end_station in bike_differences:
                    bike_differences[end_station] -= 1
                else:
                    bike_differences[end_station] = -1
                # Set action if previous node is root, otherwise use previous action
                action = end_station if node.action == NULL_STATION else node.action
                # Create node and add to expanded list
                new_node = Node(
                    action=action,
                    station=end_station,
                    mode='bike', 
                    root_mode=node.root_mode,
                    time=node.time + WALK_TIMES[node.station][end_station],
                    reward=node.reward + get_reward(incentives[end_station], 'rent'),
                    depth=node.depth + 1,
                    prev=node,
                    bike_differences=bike_differences
                )
                new_nodes.append(new_node)  
        #------------------------------------- Bike -------------------------------------------
        elif node.mode == 'bike':
            #-------------------------- Get incentives -------------------------
            # Get predicted incentives based on node time
            predicted_incentives_index = bisect.bisect_right(incentive_times, node.time) - 1
            incentive_time, incentives = predicted_incentives[predicted_incentives_index]
            incentives = incentives.copy()
            # Adjust incentives based on bike differences
            for station, bike_difference in node.bike_differences.items():
                bike_count = estimate_bike_count(station, root_bike_counts[station], incentive_time - root_time)
                bike_count += bike_difference
                incentives[station] = INCENTIVES[station][bike_count]
            #----------------------- Assess Stations ---------------------------
            # Consider neutral stations if there's a reward for renting from current station
            can_go_to_neutral = (
                # Current reward is greater than previous
                (node.prev != None and node.reward > node.prev.reward)
                # Or current station is incentivized for return
                or get_reward(incentives[node.station], 'rent') > 0
                # ToDo: Probably only need second option
            )
            # Get reward rates and travel times
            station_reward_rate_pairs = []
            station_travel_time_pairs = []
            for end_station, incentive in enumerate(incentives):
                # Skip station
                bike_time = BIKE_TIMES[node.station][end_station]
                return_time = node.time + bike_time + WALK_TIMES[end_station][self.final_station]
                if (
                    # Same station
                    end_station == node.station
                    # Return time exceeds excursion end time
                    or return_time > self.end_time
                ):
                    continue
                # Get reward rate for every incentivized stations
                if incentive > 0:
                    reward = get_reward(incentive, 'return')
                    reward_rate = reward / BIKE_TIMES[node.station][end_station]
                    station_reward_rate_pairs.append( (end_station, reward_rate) )
                # Get travel times for neutral stations
                if can_go_to_neutral and incentive == 0:
                    station_travel_time_pairs.append( (end_station, BIKE_TIMES[node.station][end_station]) )
            #---------------------- Get best stations --------------------------
            BIKE_BRANCH_FACTOR = AGENT_SEARCH_BRANCH_FACTOR
            new_stations = []
            # Get incentivized stations to expand to
            if station_reward_rate_pairs:
                # Sort stations by descending reward rate
                station_reward_rate_pairs = sorted(station_reward_rate_pairs, key=lambda x: x[1], reverse=True)
                stations = [pair[0] for pair in station_reward_rate_pairs]
                # Add stations up to branch factor
                new_stations.extend(stations[:BIKE_BRANCH_FACTOR])
            # Not enough incentivized stations, try neutral
            if can_go_to_neutral and len(new_stations) < BIKE_BRANCH_FACTOR:
                if station_travel_time_pairs:
                    # Sort stations by ascending travel time
                    station_travel_time_pairs = sorted(station_travel_time_pairs, key=lambda x: x[1])
                    # Validate and add station to new stations
                    for end_station, _ in station_travel_time_pairs:
                        trip_end_time = node.time + BIKE_TIMES[node.station][end_station]
                        end_bike_count = estimate_bike_count(end_station, root_bike_counts[end_station], trip_end_time - root_time)
                        if self._validate_station(end_station, end_bike_count, node.mode):
                            new_stations.append(end_station)
                        # Add stations until branch factor is met
                        if len(new_stations) == BIKE_BRANCH_FACTOR:
                            break
            #-------------------- Instantiate new nodes ----------------------
            for end_station in new_stations:
                # Adjust bike difference for end station
                bike_differences = node.bike_differences.copy()
                if end_station in bike_differences:
                    bike_differences[end_station] += 1
                else:
                    bike_differences[end_station] = 1
                # Set action if previous node is root, otherwise use previous action
                action = end_station if node.action == NULL_STATION else node.action
                # Create node and add to expanded list
                new_node = Node(
                    action=action,
                    station=end_station,
                    mode='walk', 
                    root_mode=node.root_mode,
                    time=node.time + BIKE_TIMES[node.station][end_station],
                    reward=node.reward + get_reward(incentives[end_station], 'return'),
                    depth=node.depth + 1,
                    prev=node,
                    bike_differences=bike_differences
                )
                new_nodes.append(new_node)
        # Return child nodes
        return new_nodes
           
    def _validate_station(self, station: int, bike_count: int, mode: str) -> bool:
        """ Returns true iff the station is a valid destination. That means going here
        won't fill up or empty the station too much, resulting in failures.
        Args:
            mode (str): {'walk', 'bike'} the mode of travel used to reach
                the given station
        """
        if mode == 'walk':
            # Remove bike, limited by capacity
            bike_count = min(bike_count + 1, CAPACITIES[station])
            # Get incentive
            incentive = INCENTIVES[station][bike_count]
            # Station is not valid if returns are incentivized
            if incentive > 0:
                return False
        else:  # mode == 'bike'
            # Add bike, limited by capacity
            bike_count = max(bike_count - 1, 0)
            # Get incentive
            incentive = INCENTIVES[station][bike_count]
            # Station is not valid if rentals are incentivized
            if incentive < 0:
                return False
        # Station is valid
        return True
        

def format_time(time: float) -> str:
    """ Takes a float representing the time of day in hours and returns
    a string in hh:mm format. 
    """
    hours = int(time)
    minutes = int((time - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"


def get_closest_stations(current_station: int, max_travel_time: float, mode: str) -> list:
    """ Returns a list of the closest stations to the current station, sorted by travel time
    for the given mode, and limited by max_travel_time.
    Args:
        current_station (int): The start station from which the function finds the closest
            stations to travel to
        max_travel_time (float): The stations in the returned list have a travel time at most
            equal to max_travel_time
        mode: ('walk', 'bike') The mode of travel from the current_station
    """
    # Get list of travel times to each station and list of stations osrted by travel time
    if mode == 'walk':
        travel_times = WALK_TIMES[current_station]
        near_stations = NEAR_WALK_STATIONS
    elif mode == 'bike':
        travel_times = BIKE_TIMES[current_station]
        near_stations = NEAR_BIKE_STATIONS
    # Binary search for cutoff
    left, right = 0, len(near_stations)
    while left < right:
        mid = (left + right) // 2
        if travel_times[near_stations[mid]] <= max_travel_time:
            left = mid + 1
        else:
            right = mid
    # Return nearest stations before time cutoff
    return near_stations[:left]


def generate_bike_counts(bike_counts: list, elapsed_time: float) -> list:
    """ Returns a generated list of bike counts for each station after the elapsed time. """
    # Don't modify given list
    counts = list(bike_counts)
    # Time must be non-negative
    if elapsed_time < 0:
        logger.error(f"{ERROR} Error: elapsed_time {elapsed_time:.4f} less than 0")
    # If no time has passed, use the same counts
    if np.isclose(0, elapsed_time):
        return counts
    # Generate change in bike count for each station
    for station, count in enumerate(counts):
        # ToDo: account for full empty stations (see retrospective_notes.md)
        # Estimate incoming/outgoing bikes
        est_bikes_in = elapsed_time * RETURN_RATES[station]
        est_bikes_out = elapsed_time * RENT_RATES[station]
        # Generate incoming/outgoing bike counts
        bikes_in = np.random.poisson(est_bikes_in)
        bikes_out = np.random.poisson(est_bikes_out)
        bike_diff = int(round(bikes_in - bikes_out))
        # Adjust bike count and limit by capacity
        count += bike_diff 
        count = max(0, min(CAPACITIES[station], count))
        # Update count
        counts[station] = count
    return counts


def estimate_bike_count(station: int, bike_count: int, time_difference: float, overestimate: bool=True) -> int:
    """ Given the current bike count of a station, returns an estimate of the 
    bike count after the time difference (or before, if time_difference < 0).
    This is a deterministic estimation and should not be used for simulation.
    Args:
        time_difference (float): amount of time in hours after (or before, if
            time_difference < 0) the station has the given bike_count.
        overestimate (bool): (experimental) rounds up estimated bike change
            to produce an optimistic estimate for the agent
    """
    # Bike count is unchanged for zero time difference
    if time_difference == 0:
        return bike_count
    # Estimate change in bike count during elapsed time
    if overestimate:
        # Use net rate and round up absolute value of bike difference
        net_rate = RETURN_RATES[station] - RENT_RATES[station]
        if net_rate > 0:
            bike_difference = np.ceil(time_difference * net_rate)
        else:
            bike_difference = np.floor(time_difference * net_rate)
    else:
        # Estimate bikes added/removed individually using floor
        bikes_added = np.floor(time_difference * RETURN_RATES[station])
        bikes_removed = np.floor(time_difference * RENT_RATES[station])
        bike_difference = bikes_added - bikes_removed
    # Subtract difference to get previous bike count
    new_bike_count = bike_count + bike_difference
    # Limit by capacity
    new_bike_count = max(0, min(CAPACITIES[station], new_bike_count))
    return int(new_bike_count)


def get_reward(incentive: float, option: str) -> int:
    """ Returns the reward for returning or renting a bike to a station
    with the given incentive. 
    Args:
        incentive (float): the incentive of the station
        option (str): {'rent', 'return'}"""
    # If incentive is for other option, reward is 0
    if (
        (incentive < 0 and option == 'return')
        or (incentive > 0 and option == 'rent')
        ):
        return 0
    # Remove sign
    reward = abs(incentive)
    # Limit reward
    # reward = min(1, reward)
    return reward


def log_bike_counts(bike_counts: list) -> None:
    """ Logs a list of every station's bike count and capacity. """
    logger.info(f'--- Bike Counts ---')
    for i in range(N):
        logger.info(f'{i}: ({bike_counts[i]}/{CAPACITIES[i]})')
    logger.info('-------------------')


def log_data_stats(data: dict) -> None:
    for key in data:
        logger.error(f'{key}:')
        logger.error(f'\tMean: {np.mean(data[key]):.2f}')
        logger.error(f'\tMedian: {np.median(data[key]):.2f}')
        logger.error(f'\tStd Dev: {np.std(data[key]):.2f}')
        logger.error(f'\tMax: {np.max(data[key]):.2f}')
        logger.error(f'\tMin: {np.min(data[key]):.2f}')


def log_incentivized_stations(incentives: list) -> None:
    """ Logs incentivized stations categorized by incentive option (rentals/returns) """
    rent_stations = []
    return_stations = []
    for i, incentive in enumerate(incentives):
        if incentive > 0:
            return_stations.append(i)
        elif incentive < 0:
            rent_stations.append(i)
    logger.info(f'Rent:    {rent_stations}\nReturn:  {return_stations}')


def simulate_bike_share(return_full_stats=False, batch_stats_only=False) -> float | dict:
    """ Simulates a single excursion of an agent rebalancing bikes in a bike share system,
    and returns a dictionary of stats including rewards earned and trips taken. 

    In the returned stats, time_left and wait_time are in minutes, real_time_duration
    is in seconds, and everything else is in hours.
    Args:
        return_full_stats (bool): Returns the mean reward if False. Returns a dict of stats
            if True.
        batch_stats_only (bool): Returns only stats for batch runs if True. Returns all
            stats if False (unless return_full_stats is False, in which case this arg
            has no effect).
    """
    # Track simulation time
    real_start_time = time.perf_counter()
    #--------------- Set seed ---------------
    global SEED
    if USE_STATIC_SEED and SIM_MODE == 'single_run':
        logger.info(f"Using fixed seed: {SEED}")
    else:
        SEED = np.random.randint(0, 2**32, dtype=np.uint32)  # Generate unsigned 32 bit int
        logger.info(f"Generated random seed: {SEED}") 
    np.random.seed(SEED)
    #------------ Initialize params/vars -----------
    # Initialize bike system
    bike_counts = generate_bike_counts(INITIAL_BIKE_COUNTS, WARM_UP_TIME)
    incentives = [INCENTIVES[station][bike_counts[station]] for station in range(N)]
    current_time = START_TIME  # Hour of the day (HH)
    # Initialize agent
    agent = Agent()
    # Initialize stats (action counts)
    bike_trip_count = 0
    walk_trip_count = 0
    wait_count = 0
    actions = []
    total_walk_time = 0
    total_bike_time = 0
    total_wait_time = 0
    total_bike_distance = 0
    #----------------------------- Start Simulation ----------------------------------
    logger.info(f'---------- Starting Excursion -----------')
    logger.info(f'Warmup time: {WARM_UP_TIME} hours\nStart time: {format_time(START_TIME)} \nExcursion time: {agent.end_time - START_TIME} hours\n')
    # Each iteration corresponds to one trip (or wait) from the agent
    has_time_for_trip = True
    while has_time_for_trip:
        # Report current state
        logger.info(f"> {agent.station} ({format_time(current_time)})")
        log_incentivized_stations(incentives)
        # Get agent destination
        end_station = agent.get_action(bike_counts, incentives, current_time)
        #-------------------- No trip found ----------------------
        # No trip found, try switching modes or wait till next update
        if end_station == NULL_STATION:
            logger.warning(f'\t{WARNING} No trip found')
            #--------------- Try walking ------------------
            # If biking, try one-off walking attempt before waiting
            if agent.mode == 'bike':
                logger.warning(f'\t{WARNING} One-off walking attempt')
                agent.mode = 'walk'
                # Look for path starting with walk action
                end_station = agent.get_action(bike_counts, incentives, current_time)
                # If not enough time to walk and bike, just end trip
                if end_station == END_TRIP:
                    logger.warning(f'\t{WARNING} One-off walking END_TRIP')
                # If walking also failed, switch mode back
                elif end_station == NULL_STATION:
                    agent.mode = 'bike'
                # If station found, process trip
                else:
                    logger.error(f'\t{WARNING} Switch to walk mode')
            #------------------- Wait ----------------------
            # If biking or one-off walking attempt failed, wait
            if end_station == NULL_STATION:
                wait_count += 1
                # Get next incentive update time
                update_time = np.ceil(current_time / INCENTIVE_UPDATE_INTERVAL) * INCENTIVE_UPDATE_INTERVAL
                # If update time is close to current time, get next update time
                # ToDo: this could introduce issue where incentive update is skipped
                if np.isclose(update_time, current_time):
                    update_time += INCENTIVE_UPDATE_INTERVAL
                # Update is after normal wait time (with leniency)
                if update_time > current_time + AGENT_WAIT_TIME + AGENT_WAIT_TIME_LENIENCY:
                    # Use normal wait time
                    wait_time = AGENT_WAIT_TIME
                    # Update bike counts
                    bike_counts = generate_bike_counts(bike_counts, wait_time)
                # Update is before normal wait time (or within leniency)
                else: 
                    # Only wait till update           
                    wait_time = update_time - current_time
                    # Update bike counts
                    bike_counts = generate_bike_counts(bike_counts, wait_time)
                    # Update incentives
                    incentives = [INCENTIVES[station][bike_counts[station]] for station in range(N)]
                # Update time and report
                current_time += wait_time
                logger.info(f'\t{MISS} No trip found. Wait until {format_time(current_time)}')
                total_wait_time += wait_time
                actions.append({
                    'end_station' : None,
                    'agent_mode' : 'wait',
                    'duration' : wait_time,
                    'rent_reward' : None,
                    'return_reward' : None,
                    'distance' : None
                }) 
                # Go to next trip
                continue
        #--------------- Out of time -------------------
        # Agent is out of time, go to final station
        if end_station == END_TRIP:
            agent.mode = 'walk'  # Hard set agent to walk
            end_station = agent.final_station
            has_time_for_trip = False
            logger.warning(f'\t{WARNING} End excursion')
        # ----------------------- Process trip ------------------------
        # Generate trip duration
        if agent.mode == 'bike':
            bike_trip_count += 1
            avg_trip_duration = BIKE_TIMES[agent.station][end_station]
        elif agent.mode == 'walk':
            walk_trip_count += 1
            avg_trip_duration = WALK_TIMES[agent.station][end_station]
        #trip_duration = np.random.exponential(avg_trip_duration) # ToDo: use better distribution
        # ToDo testing
        trip_duration = avg_trip_duration
        #---------------- Get incentives / Update bike counts -------------
        # Get time of last incentive update before trip ends
        end_time = current_time + trip_duration
        update_time = np.floor(end_time / INCENTIVE_UPDATE_INTERVAL) * INCENTIVE_UPDATE_INTERVAL
        # If last update was before start of trip, use same incentives
        if update_time < current_time:
            new_incentives = incentives
            # Update bike counts to end of trip time
            bike_counts = generate_bike_counts(bike_counts, end_time - current_time)
        # Otherwise update bike counts in two steps to get new incentives
        else:
            # Update bike counts to time of incentive update and get new incentives
            bike_counts = generate_bike_counts(bike_counts, update_time - current_time)
            new_incentives = [INCENTIVES[station][bike_counts[station]] for station in range(N)]
            # Update bike counts to time of end of trip
            bike_counts = generate_bike_counts(bike_counts, end_time - update_time)
        #------------ Update reward -------------
        # Update bike counts based on agent action, and get reward
        rent_reward = 0
        return_reward = 0
        reward = 0
        if agent.mode == 'bike':
            # Remove bike from start station
            bike_counts[agent.station] = max(0, min(CAPACITIES[agent.station], bike_counts[agent.station] - 1))
            # Add bike to end station
            bike_counts[end_station] = max(0, min(CAPACITIES[end_station], bike_counts[end_station] - 1))
            # Note: Incentives are locked in once a trip begins, so we use the original incentives.
            # Get reward
            rent_reward = get_reward(incentives[agent.station], option='rent')
            return_reward = get_reward(incentives[end_station], option='return')
            reward = rent_reward + return_reward
        #------------- Update State --------------
        # Record action for stats
        if agent.mode == 'walk':
            total_walk_time += trip_duration
            # distance = WALK_DISTANCES[agent.station][end_station]
            # total_walk_distance += distance
        elif agent.mode == 'bike':
            total_bike_time += trip_duration
            total_bike_distance += BIKE_DISTANCES[agent.station][end_station]
        actions.append({
            'end_station' : end_station,
            'agent_mode' : agent.mode,
            'duration' : trip_duration,
            'rent_reward' : rent_reward,
            'return_reward' : return_reward,
            'distance' : None
        })
        # Update incentives
        incentives = new_incentives
        # Report trip
        reward_str = ''
        if agent.mode == 'bike':
            reward_str = f'+{reward:.2f}'
        if trip_duration*60 < 10:
            duration_str = f'{trip_duration*60:.1f} min'
        else:
            duration_str = f'{round(trip_duration*60)} min'
        logger.info(f'\t{HIT} {agent.mode.capitalize()} to {end_station} ({duration_str}) {reward_str}')
        # Update agent
        agent.update(end_station, reward)
        # Update time    
        current_time = end_time
        # Report if excursion time exceeded
        time_exceeded = current_time - agent.end_time
        if time_exceeded > 0:
            logger.warning(f'\t{WARNING} End time exceeded by {time_exceeded*60:.1f} min')
        # Testing: end simulation early 
        if HARD_STOP_TIME > 0 and current_time - START_TIME > HARD_STOP_TIME:
            logger.warning(f'\t{WARNING} Simulation stopped early.')
            break
    #------------------------------- Simulation complete -----------------------------
    real_time_duration = time.perf_counter() - real_start_time
    future_system_fail_count = np.sum([FAIL_COUNTS[station][bike_counts[station]] for station in range(N)])
    # Report simulation data
    logger.info(f"> {agent.station} ({format_time(current_time)})")
    logger.info(f'\n---------- Excursion Complete ----------')
    logger.info(f'End time: {format_time(current_time)}')
    logger.info(f'Final station: {agent.station}')
    logger.info(f'Bike count: {bike_trip_count}')
    logger.info(f'Walk count: {walk_trip_count}')
    logger.info(f'Wait count: {wait_count}')
    logger.info(f'Total reward: {agent.reward}')
    logger.info(f'Future system fail count: {future_system_fail_count}')
    logger.info(f'Seed: {SEED}')
    logger.info('')
    # Return stats if prompted to
    if return_full_stats:
        data = dict()
        # Time left is in minutes, and if it's negative the agent ended late
        data['time_left'] = 60 * ((START_TIME + EXCURSION_TIME) - current_time)
        data['bike_count'] = bike_trip_count
        # Bike distance is in km
        data['bike_distance'] = 1000 * total_bike_distance 
        data['bike_time'] = total_bike_time
        data['walk_time'] = total_walk_time
        # Wait time is in minutes
        data['wait_time'] = 60 * total_wait_time
        data['reward'] = agent.reward
        data['future_system_fail_count'] = future_system_fail_count
        # Following stats not included for batch
        if not batch_stats_only:
            data['real_time_duration'] = real_time_duration
            data['final_bike_counts'] = bike_counts
            data['final_incentives'] = incentives
            data['actions'] = actions
            data['walk_count'] = walk_trip_count
            data['wait_count'] = wait_count 
        return data
    # Return agent's total excursion reward by default
    return agent.reward


def simulate_batch(batch_size: int) -> dict:
    """ Runs a batch of bike share simulations and returns stats """
    # Iterate through batch
    batch_data = None
    for i in range(batch_size):
        run_data = simulate_bike_share()
        # Create dict based on simulation data if first run
        if batch_data == None:
            batch_data = {key: [] for key in run_data}
        # Add run data to dict
        for key in run_data.keys():
            batch_data[key].append(run_data[key])
        logger.error(f'\rRunning simulation batch: {i+1}/{batch_size} complete')
    # Analyze batch data
    logger.error(f'\n------- Batch Complete ({batch_size} runs) --------')
    logger.error(f'Agent Intelligence: {AGENT_INTELLIGENCE}')
    log_data_stats(batch_data)


def generate_results_filepath(seed=None):
    """ Generates and returns resutls filepath in the format 'results/YYMMDD_HHMM_s<seed>.json' """
    results_dir = os.path.join(BASE_PATH[:-1], 'results')
    # Generate timestamp
    timestamp = time.strftime('%y%m%d_%H%M')
    if seed is None:
        seed = 'batch'
    else:
        seed = f's{seed}'
    base_name = f'{timestamp}_{seed}'
    ext = '.json'
    filename = base_name + ext
    counter = 2 
    # Increment filename if it already exists
    while os.path.exists(os.path.join(results_dir, filename)):
        if '-' in base_name:
            base_name = base_name.rsplit('-', 1)[0]  # Remove last -n if exists
        filename = f"{base_name}-{counter}{ext}"
        counter += 1
    return os.path.join(results_dir, filename)


def record_single_run_results(data: dict):
    filepath = generate_results_filepath(SEED)
    # Set seed string for report
    if USE_STATIC_SEED:
        seed_str = f'Static Seed: {SEED}'
    else:
        seed_str = f'Randomly Generated Seed: {SEED}'
    # Set trip string for report
    trip_str = ''
    time = START_TIME
    for action in data['actions']:
        if action['agent_mode'] == 'wait':
            # Report wait
            trip_str += f"\n{format_time(time)} > {action['agent_mode'].capitalize()} ({action['duration']*60:.1f} min)"
        else:
            # Report bike/walk trip
            trip_str += f"\n{format_time(time)} > {action['agent_mode'].capitalize()} to {action['end_station']} ({action['duration']*60:.1f} min)"
            # Report rewards
            if action['rent_reward'] > 0:
                trip_str += f"\n\tRental Reward: + {action['rent_reward']}"
            if action['return_reward'] > 0:
                trip_str += f"\n\tReturn Reward: + {action['return_reward']}"
        time += action['duration']
    # Determine punctuality
    punc_str = 'Perfect'
    time_left = round(data['time_left'], 2)
    if time_left > 0:
        punc_str = f'{time_left:.2f} minutes early'
    if time_left < 0:
        punc_str = f'{abs(time_left):.2f} minutes late'
    #------------ Report for frontend -----------
    report = (
        f"{filepath}\n" +
        "\nSingle Run Simulation\n" +
        f"{seed_str}\n" +
        f"Runtime: {data['real_time_duration']:.6f} seconds\n" +
        "\nParameters\n" +
        f"Agent Mode: {USER_PARAMS['agent_mode'].capitalize()}\n" +
        f"Start Station: {USER_PARAMS['start_station']}\n" +
        f"End Station: {USER_PARAMS['end_station']}\n" +
        f"Excursion Time: {USER_PARAMS['excursion_time']} hours\n" +
        f"Warmup Time: {USER_PARAMS['warmup_time']} hours\n" +
        f"Empty Station Bias: {USER_PARAMS['empty_bias']}\n" +
        f"Full Station Bias: {USER_PARAMS['full_bias']}\n" +
        "\nResults\n" +
        f"Total Reward: {data['reward']}\n" +
        f"Punctuality: {punc_str}\n" +
        f"Bikes Rented: {data['bike_count']}\n" +
        f"Total Distance Biked: {data['bike_distance']:.2f} km\n" +
        f"Total Time Biking: {data['bike_time']:.2f} hours\n" +
        f"Total Time Walking: {data['walk_time']:.2f} hours\n" +
        f"Total Time Waiting: {data['wait_time']:.2f} minutes\n" +
        f"Expected Future Failures: {data['future_system_fail_count']:.2f}\n" +
        f"\nAgent Actions {trip_str}"
    )
    print(report)
    results = {
        'data' : data,  # Raw sim results
        'user_parms' : USER_PARAMS,  # User params
        'report' : report,  # Text for frontend
    }
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=2)
    print(filepath)


def record_batch_precision_results(results):
    filepath = generate_results_filepath()
    means, deltas, replication_count, runtime = results
    # Determine punctuality
    punc_str = 'Perfect'
    time_left = means['time_left']
    if time_left > 0:
        punc_str = f"{time_left} ± {deltas['time_left']} minutes early"
    if time_left < 0:
        punc_str = f"{abs(time_left)} ± {deltas['time_left']} minutes late"
    # Determine if max time exceeded
    time_exceeded_str = ''
    if runtime > USER_PARAMS['max_runtime']:
        time_exceeded_str = 'Warning: Max runtime exceeded. Target precision for result estimations not reached.\n'
    report = (
        f"{filepath}\n" +
        "\nBatch Simulation\n" +
        f"Replication Count: {replication_count}\n" +
        f"Total Runtime: {runtime:.6f} seconds\n" +
        time_exceeded_str +
        "\nParameters\n" +
        # Run parameters
        f"Agent Mode: {USER_PARAMS['agent_mode'].capitalize()}\n" +
        f"Start Station: {USER_PARAMS['start_station']}\n" +
        f"End Station: {USER_PARAMS['end_station']}\n" +
        f"Excursion Time: {USER_PARAMS['excursion_time']} hours\n" +
        f"Warmup Time: {USER_PARAMS['warmup_time']} hours\n" +
        f"Empty Station Bias: {USER_PARAMS['empty_bias']}\n" +
        f"Full Station Bias: {USER_PARAMS['full_bias']}\n" +
        # Batch parameters
        f"Confidence Level: {USER_PARAMS['confidence_level']}\n" +
        f"Parallel Batch Size: {USER_PARAMS['parallel_batch_size']}\n" +
        # Precision parameters
        f"Min Sample size: {USER_PARAMS['min_sample_size']} runs\n" +
        "\nResults\n" +
        '(Expected values for a single run)\n' +
        f"Total Reward: {means['reward']} ± {deltas['reward']}\n" +
        f"Punctuality: {punc_str}\n" +
        f"Bikes Rented: {means['bike_count']} ± {deltas['bike_count']}\n" +
        f"Total Distance Biked: {means['walk_time']} ± {deltas['walk_time']} km\n" +
        f"Total Time Biking: {means['bike_time']} ± {deltas['bike_time']}\n" +
        f"Total Time Walking: {means['walk_time']} ± {deltas['walk_time']} hours\n" +
        f"Total Time Waiting: {means['wait_time']} ± {deltas['wait_time']} minutes\n" +
        f"Expected Future Failures: {means['future_system_fail_count']} ± {deltas['future_system_fail_count']}\n"
    )
    print(report)
    results = {
        'data' : results,  # Raw sim results
        'user_parms' : USER_PARAMS,  # User params
        'report' : report,  # Text for frontend
    }
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=2)
    print(filepath)
    
    
def record_batch_fixed_results():
    pass

    
def main():    
    
    print()
    
    if SIM_MODE == 'single_run':
        logger.setLevel(SINGLE_RUN_LOG_LEVEL)
        results = simulate_bike_share(True)
        record_single_run_results(results)
        
    elif SIM_MODE == 'batch':
        logger.setLevel(BATCH_LOG_LEVEL)
        if BATCH_MODE == 'precision_based':
            results = estimate_stochastic_stats(
                process=simulate_bike_share, 
                args=(True, True),
                min_samples=MIN_SAMPLE_SIZE,
                max_runtime=MAX_RUNTIME,
                relative_margin_of_error=RELATIVE_MARGIN_OF_ERROR,
                minimum_margin_of_error=ABSOLUTE_MARGIN_OF_ERROR,
                confidence_level=CONFIDENCE_LEVEL, 
                batch_size=PARALLEL_BATCH_SIZE,
                log_progress=PRINT_BATCH_PROGRESS
            )
            record_batch_precision_results(results)
        elif BATCH_MODE == 'fixed_sample_size':
            simulate_batch(BATCH_SIZE)

    file_handler.close()
    logging.shutdown()
    

if __name__ == "__main__":
    main()