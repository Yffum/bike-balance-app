# Main simulation
import json
from dataclasses import dataclass, field
import logging
import time
import os
from queue import PriorityQueue
import bisect
import sys
import io
import re

import numpy as np

from tools import estimate_stochastic_stats, estimate_stochastic_stats_fixed_size
        
# Unless otherwise indicated:
# - "station" refers to station index
# - "time" refers to simulated time measured in hours

# Establish data directory for dev builds
if os.path.exists('external'):
    BASE_PATH = 'external/'
else:
    BASE_PATH = ''

# Check if called from app
USING_APP = False
if len(sys.argv) == 2 and sys.argv[1] == '--frontend':
    USING_APP = True
    
#=============================== STATIC PARAMETERS ===========================================

#-------------------- CONFIGURATION --------------------
START_TIME = 16.0  # Time of day (HH) when the simulation begins                                                     

# The number of bikes to add/remove when validating a station (higher
# values are stricter, while 0 means the station is valid if there is
# no inverse incentive)
AGENT_VALIDATION_EXTENT = 2 
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
BIKE_TIME_CV = 0.1     # Coefficient of variation for bike time lognormal RV generation
WALK_TIME_CV = 0.1     # Coefficient of variation for walk time lognormal RV generation


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
RANDOMIZE_STATIONS = USER_PARAMS['randomize_stations'] # If True, randomizes start/end stations
BATCH_MODE = USER_PARAMS['batch_mode']  # 'precision_based' or 'fixed_sample_size'
# - Fixed sample size
FIXED_SAMPLE_SIZE = USER_PARAMS['batch_size']  # Number of simulation replications
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
PRINT_BATCH_PROGRESS = False
# Time range for debug log (HH)
DEBUG_START_TIME = 17 + 20/60
DEBUG_END_TIME = 17 + 30/60

WRITE_LOG_FILE = False

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(SINGLE_RUN_LOG_LEVEL) 
# Create string buffer for single run report
LOG_STREAM = io.StringIO()
EXCURSION_DELIMITER = '$'  # Marks beginning and end of excursion in log

# If script called from frontend
if USING_APP:
    PRINT_BATCH_PROGRESS = False
    # Use log stream for single run report
    if SIM_MODE == 'single_run':
        stream_handler = logging.StreamHandler(LOG_STREAM)
        stream_handler.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)
    # Disable log for batch
    else:
        logging.disable()
# Script called directly, use console stream
else:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    
def generate_log_filepath(seed: int | str) -> str:
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

if WRITE_LOG_FILE:
    filepath = generate_log_filepath(SEED)
    file_handler = logging.FileHandler(filepath, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

#----------------------- Set Up Output Directories ---------------------
path = BASE_PATH + 'results'
if not os.path.exists(path):
    logger.info('Results folder not found.')
    os.makedirs(path)
    logger.info(f'{path} folder created.')
path = BASE_PATH + 'logs'
if not os.path.exists(path):
    logger.info('Logs folder not found.')
    os.makedirs(path)
    logger.info(f'{path} folder created.')


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
    def __init__(self, start_station: int, final_station: int):
        """ An agent iteratively takes trips between stations, alternating between biking and walking. """
        self.station = start_station  # Current station index
        self.final_station = final_station  # The station the agent wants to end at when time is done
        self.mode = 'bike'
        self.end_time = START_TIME + EXCURSION_TIME  # Agent should arrive at final station by this time
        self.reward = 0  # Total rewards earned
        self.can_rent_bike = True # True if the agent cant rent a bike from the current station
                                  # (becomes False after delivering a bike)
        self._is_ending = False

    def update(self, new_station: int, reward_gain: float) -> None:
        """ Updates the agent's state based on the given trip data. 
        Args:
            new_station (int): The station the agent just traveled to
            reward_gain (float): The amount of reward gained from the
                trip the agent just took. """
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
        """ Returns the next station the agent will travel to. 
        Args:
            bike_counts (list): A list of current bike counts indexed
                by station
            incentives (list): A list of current incentives indexed by
                station
            current_time (float): The current time in the simulation.
        """
        # If ending excursion, find best path home
        if self._is_ending == True:
            return self._get_ending_action(bike_counts)
        # Use smart action if agent is smart, otherwise use basic action
        elif AGENT_INTELLIGENCE == 'smart':
            return self._get_smart_action(bike_counts, incentives, current_time)
        elif AGENT_INTELLIGENCE == 'basic':
            test =  self._get_basic_action(bike_counts, incentives, current_time)
            return test

    def _get_basic_action(self, bike_counts: list, incentives: list, current_time: float) -> int:
        """" Returns the next station the agent will travel without prediction. 
        Args:
            bike_counts (list): A list of current bike counts indexed
                by station
            incentives (list): A list of current incentives indexed by
                station
            current_time (float): The current time in the simulation
        """
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
            # Validate station
            if not self._validate_station(
                end_station, bike_counts[end_station], self.mode
                ):
                continue
            # Ensure there is enough time to reach final station
            return_time = (
                current_time 
                + trip_times[self.station][end_station]
                + self._estimate_time_to_end_excursion_precise(bike_counts, end_station)
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
            self._is_ending = True
            return self._get_ending_action(bike_counts)
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
                        + self._estimate_time_to_end_excursion_precise(bike_counts, self.station)
                    )
                    if return_time > self.end_time:
                        self._is_ending = True
                        return self._get_ending_action(bike_counts)
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
        """ Returns the next station the agent will travel to using a search tree. 
        Args:
            bike_counts (list): A list of current bike counts indexed
                by station
            incentives (list): A list of current incentives indexed by
                station
            current_time (float): The current time in the simulation.
        """
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
                        return_time_after_update = update_time + self._estimate_time_to_end_excursion(self.station)
                        # Wait for update if there's time and there
                        # are no incentivized stations
                        if (
                            return_time_after_update < self.end_time
                            and all(x == 0 for x in incentives)
                        ):
                            return NULL_STATION   
                        #------------ Otherwise end trip ------------
                        else:
                            logger.debug(f'eta: {format_time(return_time_after_update)}, root time: {format_time(root_time)}')
                            self._is_ending = True
                            return self._get_ending_action(bike_counts)
        #--------------------- Best node found ------------------------
        if best_node.action != NULL_STATION:
            # Return best action
            return best_node.action
        #--------------- No best node, end excursion ------------------
        # If no best node was found, then agent doesn't have time left
        else:
            self._is_ending = True
            return self._get_ending_action(bike_counts)  # Finish excursion       

    def _expand_node(self, node: Node, root_bike_counts: list, root_time: float, predicted_incentives: list) -> list:
        """ Expands the given search node and returns its children.
        Args:
            node (Node): The node to expand.
            root_bike_counts (list): The bike counts of the simulation
                at the time of the root node of the search tree
            root_time (float): The simulation time of the root node of
                the search tree
            predicted_incentives (list): The incentives predicted at the
                time of the given node to expand.
        Returns:
            list[Node]: The child nodes of the given node.
        """
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
                return_time = node.time + walk_time + self._estimate_time_to_end_excursion(end_station)
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
                if end_station == 27:
                    pass
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
                # Skip same station
                if end_station == node.station:
                    continue
                # Skip station if return time exceeds end time
                bike_time = BIKE_TIMES[node.station][end_station]
                return_time = node.time + bike_time + self._estimate_time_to_end_excursion(end_station)
                if return_time > self.end_time:
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
            station (int): the station being traveled to
            bike_count (int): the bike count of the station at the time
                it is reached
            mode (str): {'walk', 'bike'} the mode of travel used to reach
                the given station
        """
        if mode == 'walk':
            # Remove bike, limited by capacity
            bike_count = min(bike_count + AGENT_VALIDATION_EXTENT, CAPACITIES[station])
            # Get incentive
            incentive = INCENTIVES[station][bike_count]
            # Station is not valid if returns are incentivized
            if incentive > 0:
                return False
        else:  # mode == 'bike'
            # Add bike, limited by capacity
            bike_count = max(bike_count - AGENT_VALIDATION_EXTENT, 0)
            # Get incentive
            incentive = INCENTIVES[station][bike_count]
            # Station is not valid if rentals are incentivized
            if incentive < 0:
                return False
        # Station is valid
        return True
    
    def _get_ending_action(
        self, 
        bike_counts: list, 
        current_station: int=None,
        estimate_time_instead: bool=False,
        ) -> int | float:
        """ Returns the index of the station the agent should travel to
        when ending the excursion, or if estimate_time_instead is True,
        it returns the estimated travel time to end the excursion.
        Args:
            bike_counts (list): _description_
            incentives (list): _description_
            current_time (float): _description_
            estimate_time_instead (bool, optional): If True, the
                function returns the estimated travel time to end the
                excrusion instead of the next station. Defaults to
                False.
        Returns:
            (int | float): the action the agent should take, or if
                estimate_time_instead is True, the travel time to end
                the excursion
        """
        if not estimate_time_instead:
            logger.info(f'\t{WARNING} Ending excursion')
        if current_station == None:
            current_station = self.station
        # The number of available docks predicted for a station to be
        # considered a valid bike destination
        BIKE_COUNT_BUFFER = 2
        # ToDo: test this case
        # End excursion if at final station
        if current_station == self.final_station:
            if estimate_time_instead:
                return 0
            return END_TRIP
        best_action = NULL_STATION  # The action the agent should take
        # Can rent from current station
        if self.can_rent_bike:
            self.mode = 'bike'
            # Validate final station
            travel_time = BIKE_TIMES[current_station][self.final_station]
            end_bike_count = estimate_bike_count(
                self.final_station, bike_counts[self.final_station], travel_time)
            # Bike to final station if there are open docks
            if end_bike_count <= CAPACITIES[self.final_station] - BIKE_COUNT_BUFFER:
                if estimate_time_instead:
                    return BIKE_TIMES[current_station][self.final_station]
                return self.final_station
            # Can't bike to final station
            # Look for station to bike to that's closer to final station
            for near_station in NEAR_BIKE_STATIONS[self.final_station]:
                # Stop if all closer stations searched
                if near_station == current_station:
                    break
                # Validate station
                travel_time = BIKE_TIMES[current_station][near_station]
                end_bike_count = estimate_bike_count(
                    near_station, bike_counts[near_station], travel_time)
                # Bike to closest valid station
                if end_bike_count <= CAPACITIES[near_station] - BIKE_COUNT_BUFFER:
                    if estimate_time_instead:
                        # Time to bike to near station and walk to final
                        return (
                            BIKE_TIMES[current_station][near_station]
                            + WALK_TIMES[near_station][self.final_station]
                        )
                    return near_station
            # No valid station found, walk to final station
            self.mode = 'walk'
            if estimate_time_instead:
                return WALK_TIMES[current_station][self.final_station]
            return self.final_station
        # Cannot rent from current station
        else:
            self.mode = 'walk'
            direct_walk_time = WALK_TIMES[self.station][self.final_station]
            # Find nearest valid station to rent bike from
            for near_station in NEAR_WALK_STATIONS[current_station]:
                # If final station is near, just walk there
                if near_station == self.final_station:
                    if estimate_time_instead:
                        return WALK_TIMES[current_station][self.final_station]
                    return self.final_station
                # Validate station
                travel_time = WALK_TIMES[current_station][near_station]
                end_bike_count = estimate_bike_count(
                    near_station, bike_counts[near_station], travel_time)
                # Valid station to walk to and rent from found
                if end_bike_count >= 0 + BIKE_COUNT_BUFFER:
                    # Validate final station
                    travel_time2 = BIKE_TIMES[near_station][self.final_station]
                    end_bike_count = estimate_bike_count(
                        self.final_station,
                        bike_counts[self.final_station],
                        travel_time + travel_time2
                    )
                    # If walking directly is faster, do that
                    if direct_walk_time <= travel_time + travel_time2:
                        if estimate_time_instead:
                            return direct_walk_time
                        return self.final_station
                    # Walk to near station and bike to final station
                    if end_bike_count <= CAPACITIES[self.final_station] - BIKE_COUNT_BUFFER:
                        if estimate_time_instead:
                            return travel_time + travel_time2
                        return near_station
                    # Can't bike to final station
                    # Check for stations near final station
                    for near_station2 in NEAR_WALK_STATIONS[self.final_station]:
                        # No near station found, just walk to final station
                        if near_station2 == near_station or near_station2 == current_station:
                            if estimate_time_instead:
                                return WALK_TIMES[current_station][self.final_station]
                            return self.final_station
                        # Validate near_station2
                        travel_time2 = BIKE_TIMES[near_station][near_station2]
                        end_bike_count = estimate_bike_count(
                            near_station2,
                            bike_counts[near_station2],
                            travel_time + travel_time2
                        )
                        # near_station2 is valid
                        if end_bike_count <= CAPACITIES[near_station2] - BIKE_COUNT_BUFFER:
                            # Compare travel time with walking
                            total_travel_time = (
                                travel_time
                                + travel_time2
                                + WALK_TIMES[near_station2][self.final_station]
                            )
                            # If walking directly is faster, just walk
                            if direct_walk_time <= total_travel_time:
                                if estimate_time_instead:
                                    direct_walk_time
                                return self.final_station
                            # Walk to near_station, bike to near_station2, walk to final
                            if estimate_time_instead:
                                return total_travel_time
                            return near_station
        # No trip found, just walk to final station
        self.mode = 'walk'
        if estimate_time_instead:
            return WALK_TIMES[current_station][self.final_station]
        return self.final_station
            
    def _estimate_time_to_end_excursion_precise(self, bike_counts: list, current_station: int) -> float:
        """ Returns the estimated time it would take to end the excursion.
        Args:
            bike_counts (list): current bike counts indexed by station
            current_station (int): the station from which the agent
                starts ending its excursion
        """
        estimated_time = self._get_ending_action(
            bike_counts, current_station=current_station, estimate_time_instead=True)
        return estimated_time
    
    def _estimate_time_to_end_excursion(self, current_station: int) -> float:
        """ Returns a rough estimate of the time it would take to end the
        excursion from the current_station
        Args:
            current_station (int): the station from which the agent
                starts ending its excursion
        """
        ESTIMATE_BUFFER = 2/60
        return BIKE_TIMES[current_station][self.final_station] + ESTIMATE_BUFFER
    
        
#---------------------------- Tools ------------------------------------

def format_time(time: float) -> str:
    """ Takes a float representing the time of day in hours and returns
    a string in hh:mm format. 
    """
    hours = int(time)
    minutes = int((time - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"


def generate_bike_counts(bike_counts: list, elapsed_time: float) -> list:
    """ Returns a generated list of bike counts for each station after the elapsed time.
    Args:
        bike_counts (list): The previous bike counts indexed by station
        elapsed_time (float): The amount of time that has passed since
            the time of the previous bike counts, bike_counts
    """
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
        station (int): The station to estimate the bike count of
        bike_count (int): The previous bike count of the station
        time_difference (float): Amount of time in hours after (or before, if
            time_difference < 0) the station has the given bike_count.
        overestimate (bool): (experimental) Rounds up estimated bike change
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


#----------------------- Logging Functions -----------------------------

def log_bike_counts(bike_counts: list) -> None:
    """ Logs a list of every station's bike count and capacity. """
    logger.info(f'--- Bike Counts ---')
    for i in range(N):
        logger.info(f'{i}: ({bike_counts[i]}/{CAPACITIES[i]})')
    logger.info('-------------------')


def log_data_stats(data: dict) -> None:
    """ Logs standard statistics for each variable (key) in the given 
    dict.
    Args:
        data (dict): Each key corresponds to a variable e.g. runtime,
            and the value is a list of numeric data points 
    """
    for key in data:
        logger.error(f'{key}:')
        logger.error(f'\tMean: {np.mean(data[key]):.2f}')
        logger.error(f'\tMedian: {np.median(data[key]):.2f}')
        logger.error(f'\tStd Dev: {np.std(data[key]):.2f}')
        logger.error(f'\tMax: {np.max(data[key]):.2f}')
        logger.error(f'\tMin: {np.min(data[key]):.2f}')


def log_incentivized_stations(incentives: list) -> None:
    """ Logs incentivized stations categorized by incentive option (rentals/returns) 
    Args:
        incentives (list): Current incentives indexed by station
    """
    rent_stations = []
    return_stations = []
    for i, incentive in enumerate(incentives):
        if incentive > 0:
            return_stations.append(i)
        elif incentive < 0:
            rent_stations.append(i)
    logger.info(f'Rent:    {rent_stations}\nReturn:  {return_stations}')


#-------------------------- Simulation ---------------------------------

def simulate_bike_share(return_full_stats=False, batch_stats_only=False, randomize_stations=False) -> float | dict:
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
        randomize_stations (boole): Randomly select start/final stations
            for agent if True. Otherwise, use user parameters.
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
    if randomize_stations:
        agent_start_station = np.random.randint(N)
        agent_end_station = np.random.randint(N)
    else:
        agent_start_station = START_STATION
        agent_end_station = FINAL_STATION
    agent = Agent(agent_start_station, agent_end_station)
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
    logger.info(EXCURSION_DELIMITER)
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
                    'start_station' : None,
                    'end_station' : agent.station,
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
            # No more trips after this
            has_time_for_trip = False
            # If already at final station, end excursion immediately
            if agent.station == agent.final_station:
                logger.warning(f'\t{WARNING} Agent out of time, excursion ended.')
                break
            # Otherwise walk to final station
            else:
                agent.mode = 'walk'  # Hard set agent to walk
                end_station = agent.final_station
                logger.warning(f'\t{WARNING} Agent out of time, ending excursion.')
        # ----------------------- Process trip ------------------------
        # Generate trip duration
        if agent.mode == 'bike':
            bike_trip_count += 1
            mean = BIKE_TIMES[agent.station][end_station]
            stdv = BIKE_TIME_CV * mean
        elif agent.mode == 'walk':
            walk_trip_count += 1
            mean = WALK_TIMES[agent.station][end_station]
            stdv = WALK_TIME_CV * mean
        # Ensure nonzero trip time
        if mean == 0:
            logger.error(f'{ERROR} Agent {agent.mode}s from {agent.station} to {end_station}, trip time is 0')
            trip_duration = 0
        # Generate lognormal trip duration
        else:
            mu = np.log( mean**2 / np.sqrt(mean**2 + stdv**2) )
            sigma = np.sqrt(np.log(1 + stdv**2 / mean**2))
            trip_duration = np.random.lognormal(mu, sigma)
            
        # ToDo: Could cache mu/sigma for every trip duration
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
            bike_counts[end_station] = max(0, min(CAPACITIES[end_station], bike_counts[end_station] + 1))
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
            'start_station' : agent.station,
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
    logger.info(EXCURSION_DELIMITER)
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
        data['bike_distance'] = total_bike_distance / 1000
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


#------------------------ Record Results -------------------------------

def generate_results_filepath(timestamp: str, seed=None):
    """ Generates and returns results filepath in the format 'results/YYMMDD_HHMM_s<seed>.json' """
    results_dir = os.path.join(BASE_PATH[:-1], 'results')
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
        filename = f"{base_name}-{counter}{ext}"
        counter += 1
    return os.path.join(results_dir, filename)


def record_single_run_results(data: dict) -> str:
    """ Takes the results from simulate_bike_share(True) and generates
    a results file. """
    timestamp = time.strftime("%y%m%d-%H%M")
    report_date = time.strftime("%Y-%m-%d")
    report_time = time.strftime("%H:%M")
    filepath = generate_results_filepath(timestamp, SEED)
    # Set seed string for report
    if USE_STATIC_SEED:
        seed_name_value_pair = ('Static Seed', SEED)
    else:
        seed_name_value_pair = ('Randomly Generated Seed', SEED)
    # Set trip string for report, and get station visits
    trip_str = ''
    rent_counts = [0] * N  # Indexed by station
    return_counts = [0] * N
    agent_time = START_TIME
    prev_station = START_STATION
    for action in data['actions']:
        if action['agent_mode'] == 'wait':
            # Report wait
            trip_str += f"\n{format_time(agent_time)}, {action['agent_mode'].capitalize()}   ({action['duration']*60:.1f} min)"
        else:
            # Report bike/walk trip
            trip_str += f"\n{format_time(agent_time)}, {action['agent_mode'].capitalize()} to {action['end_station']}   ({action['duration']*60:.1f} min)"
            # Report rewards
            if action['rent_reward'] > 0:
                trip_str += f"\n\tRental Reward: + {action['rent_reward']}"
            if action['return_reward'] > 0:
                trip_str += f"\n\tReturn Reward: + {action['return_reward']}"
            # Count station visit
            if action['agent_mode'] == 'bike':
                rent_counts[prev_station] += 1
                return_counts[action['end_station']] += 1
        trip_str += "\n"
        agent_time += action['duration']
        prev_station = action['end_station']
    data['rent_counts'] = rent_counts
    data['return_counts'] = return_counts
    # Determine punctuality
    punc_str = 'Perfect'
    time_left = round(data['time_left'], 2)
    if time_left > 0:
        punc_str = f'{time_left:.2f} minutes early'
    if time_left < 0:
        punc_str = f'{abs(time_left):.2f} minutes late'
    #------------ Report for frontend -----------
    # Use BBCode markdown format for Godot rich text label

    sim_stat_pairs = [
        ("Date", report_date),
        ("Time", report_time),
        seed_name_value_pair,
        ("Runtime", f"{data['real_time_duration']:.5f} seconds")
    ]
    parameter_pairs = [
        ("Agent Mode", USER_PARAMS['agent_mode'].capitalize()),
        ("Start Station", USER_PARAMS['start_station']),
        ("End Station", USER_PARAMS['end_station']),
        ("Excursion Time", f"{USER_PARAMS['excursion_time']} hours"),
        ("Warmup Time", f"{USER_PARAMS['warmup_time']} hours"),
        ("Empty Station Bias", USER_PARAMS['empty_bias']),
        ("Full Station Bias", USER_PARAMS['full_bias'])
    ]
    result_pairs = [
        ("Total Reward", f"{data['reward']:.2f}"),
        ("Punctuality", punc_str),
        ("Bikes Rented", data['bike_count']),
        ("Total Distance Biked", f"{data['bike_distance']:.2f} km"),
        ("Total Time Biking", f"{data['bike_time']:.2f} hours"),
        ("Total Time Walking", f"{data['walk_time']:.2f} hours"),
        ("Total Time Waiting", f"{data['wait_time']:.2f} minutes"),
        ("Expected Future Failures", f"{data['future_system_fail_count']:.2f}")
    ]
    
    # Get excursion log with list of actions
    excursion_str = ''
    if USING_APP:
        log_str = LOG_STREAM.getvalue()
        delim = re.escape(EXCURSION_DELIMITER)
        pattern = rf"{delim}([^{delim}]+){delim}"
        match = re.search(pattern, log_str)
        if match:
            excursion_str = match.group(1)
        else:
            logger.error(f"{ERROR} No excursion delimiter pair ('{EXCURSION_DELIMITER} ... {EXCURSION_DELIMITER}') found in log stream.")
    
    report = (
        get_bbc_header("Single Run Simulation", space_above=False)
        + f"[center]{filepath}[/center]\n"
        + get_bbc_table(sim_stat_pairs)
        + get_bbc_header("Parameters")
        + get_bbc_table(parameter_pairs)
        + get_bbc_header("Results")
        + get_bbc_table(result_pairs)
        + get_bbc_header("Agent Actions", space_below=False)
        + excursion_str
    )

    # Save results
    results = {
        'data' : data,  # Raw sim results
        'date' : report_date,
        'time' : report_time,
        'user_params' : USER_PARAMS,  # User params
        'report' : report,  # BBCode for frontend
    }
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=2)
    return filepath


def record_batch_precision_results(results) -> str:
    """ Takes the results from tools.estimate_stochastic_stats and
    generates a results file. """
    timestamp = time.strftime("%y%m%d-%H%M")
    report_date = time.strftime("%Y-%m-%d")
    report_time = time.strftime("%H:%M")
    filepath = generate_results_filepath(timestamp)
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
        time_exceeded_str = '\nWarning: Max runtime exceeded. Target precision for result estimations not reached.\n'
    # Get start/end station strings 
    if RANDOMIZE_STATIONS:
        start_station_str = 'Random'
        end_station_str = 'Random'
    else:
        start_station_str = USER_PARAMS['start_station']
        end_station_str = USER_PARAMS['end_station']
    
    sim_stat_pairs = [
        ("Date", report_date),
        ("Time", report_time),
        ("Replication Count", replication_count),
        ("Total Runtime", f"{runtime:.5f} seconds"),
    ]
    parameter_pairs = [
        ("Agent Mode", USER_PARAMS['agent_mode'].capitalize()),
        ("Start Station", start_station_str),
        ("End Station", end_station_str),
        ("Excursion Time", f"{USER_PARAMS['excursion_time']} hours"),
        ("Warmup Time", f"{USER_PARAMS['warmup_time']} hours"),
        ("Empty Station Bias", USER_PARAMS['empty_bias']),
        ("Full Station Bias", USER_PARAMS['full_bias']),
        # Batch parameters
        ("Confidence Level", USER_PARAMS['confidence_level']),
        ("Parallel Batch Size", USER_PARAMS['parallel_batch_size']),
        ("Min Sample Size", f"{USER_PARAMS['min_sample_size']} runs"),
    ]
    result_pairs = [
        ("Total Reward", f"{means['reward']} ± {deltas['reward']}"),
        ("Punctuality", punc_str),
        ("Bikes Rented", f"{means['bike_count']} ± {deltas['bike_count']}"),
        ("Total Bike Distance", f"{means['bike_distance']} ± {deltas['bike_distance']} km"),
        ("Total Bike Time", f"{means['bike_time']} ± {deltas['bike_time']} hours"),
        ("Total Walk Time", f"{means['walk_time']} ± {deltas['walk_time']} hours"),
        ("Total Wait Time", f"{means['wait_time']} ± {deltas['wait_time']} minutes"),
        ("Expected Future Failures", f"{means['future_system_fail_count']} ± {deltas['future_system_fail_count']}"),
    ]
    report = (
        get_bbc_header("Batch Simulation (precision-based)", space_above=False)
        + f"[center]{filepath}[/center]\n"
        + get_bbc_table(sim_stat_pairs)
        + f"{time_exceeded_str}"  
        + get_bbc_header("Parameters")
        + get_bbc_table(parameter_pairs)
        + get_bbc_header("Results")
        + f"[center](Expected values for a single run)[/center]\n"
        + get_bbc_table(result_pairs)
    )
    
    # Convert infinite deltas to null
    for key in results[1]:
        if results[1][key] == float('inf'):
            results [1][key] = None
    # Save results
    results = {
        'data' : results,  # Raw sim results
        'date' : report_date,
        'time' : report_time,
        'user_params' : USER_PARAMS,  # User params
        'report' : report,  # Text for frontend
    }
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=2)
    return filepath
    
    
def record_batch_fixed_results(results) -> str:
    """ Takes the results from tools.estimate_stochastic_stats_fixed and
    generates a results file. """
    timestamp = time.strftime("%y%m%d-%H%M")
    report_date = time.strftime("%Y-%m-%d")
    report_time = time.strftime("%H:%M")
    filepath = generate_results_filepath(timestamp)
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
    if runtime > USER_PARAMS['max_runtime'] and replication_count < USER_PARAMS['batch_size']:
        time_exceeded_str = '\nWarning: Max runtime exceeded. Target sample size not reached.\n'
    # Get start/end station strings 
    if RANDOMIZE_STATIONS:
        start_station_str = 'Random'
        end_station_str = 'Random'
    else:
        start_station_str = USER_PARAMS['start_station']
        end_station_str = USER_PARAMS['end_station']
    
    sim_stat_pairs = [
        ("Date", report_date),
        ("Time", report_time),
        ("Replication Count", replication_count),
        ("Total Runtime", f"{runtime:.5f} seconds"),
    ]
    parameter_pairs = [
        ("Agent Mode", USER_PARAMS['agent_mode'].capitalize()),
        ("Start Station", start_station_str),
        ("End Station", end_station_str),
        ("Excursion Time", f"{USER_PARAMS['excursion_time']} hours"),
        ("Warmup Time", f"{USER_PARAMS['warmup_time']} hours"),
        ("Empty Station Bias", USER_PARAMS['empty_bias']),
        ("Full Station Bias", USER_PARAMS['full_bias']),
        # Batch parameters
        ("Confidence Level", USER_PARAMS['confidence_level']),
        ("Parallel Batch Size", USER_PARAMS['parallel_batch_size']),
    ]
    result_pairs = [
        ("Total Reward", f"{means['reward']} ± {deltas['reward']}"),
        ("Punctuality", punc_str),
        ("Bikes Rented", f"{means['bike_count']} ± {deltas['bike_count']}"),
        ("Total Bike Distance", f"{means['bike_distance']} ± {deltas['bike_distance']} km"),
        ("Total Bike Time", f"{means['bike_time']} ± {deltas['bike_time']} hours"),
        ("Total Walk Time", f"{means['walk_time']} ± {deltas['walk_time']} hours"),
        ("Total Wait Time", f"{means['wait_time']} ± {deltas['wait_time']} minutes"),
        ("Expected Future Failures", f"{means['future_system_fail_count']} ± {deltas['future_system_fail_count']}"),
    ]
    report = (
        get_bbc_header("Batch Simulation (fixed sample size)", space_above=False)
        + f"[center]{filepath}[/center]\n"
        + get_bbc_table(sim_stat_pairs)
        + f"{time_exceeded_str}"  
        + get_bbc_header("Parameters")
        + get_bbc_table(parameter_pairs)
        + get_bbc_header("Results")
        + f"[center](Expected values for a single run)[/center]\n"
        + get_bbc_table(result_pairs)
    )
    
    # Convert infinite deltas to null
    for key in results[1]:
        if results[1][key] == float('inf'):
            results [1][key] = None
    # Save results
    results = {
        'data' : results,  # Raw sim results
        'data' : report_date,
        'time' : report_time,
        'user_params' : USER_PARAMS,  # User params
        'report' : report,  # Text for frontend
    }
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=2)
    return filepath


def get_bbc_header(header: str, space_above=True, space_below=True) -> str:
    """ Returns a string for the given header, formatted in BBCode.
    Args:
        header (str): the text for the header
        space_above (bool): if True add leading newline
        space_below (bool): if True add trailing newline
    """
    text = ''
    if space_above:
        text += "\n"
    # Outline to make bold
    boldness = 1
    # Font bt is bottom spacing (number of pixels padded below header, not including trailing newline)
    lower_padding = 5
    text += f"[center][outline_size={boldness}][outline_color=white][font bt={lower_padding}]{header}[/font][/outline_color][/outline_size][/center]"
    if space_below:
        text += "\n"
    return text
    
    
def get_bbc_table(name_value_pairs) -> str:
    """ Takes a list of name-value tuples, and returns the string for
    a two column BBCode table. """
    text = ''
    text += "[table=2]\n"
    for name, value in name_value_pairs:
        text += f"[cell]{name}[/cell][cell padding=50,0,0,0]{value}[/cell]\n"
    text += "[/table]\n"
    return text


#---------------------------- Main -------------------------------------
    
def main():
    results_filepath = '' # Results filepath is printed for frontend
    
    # Based on mode, simulate bike share and save results to file
    
    if SIM_MODE == 'single_run':
        logger.setLevel(SINGLE_RUN_LOG_LEVEL)
        results = simulate_bike_share(True)
        results_filepath = record_single_run_results(results)
        
    elif SIM_MODE == 'batch':
        logger.setLevel(BATCH_LOG_LEVEL)
        
        if BATCH_MODE == 'precision_based':
            results = estimate_stochastic_stats(
                process=simulate_bike_share, 
                args=(True, True, RANDOMIZE_STATIONS),
                min_samples=MIN_SAMPLE_SIZE,
                max_runtime=MAX_RUNTIME,
                relative_margin_of_error=RELATIVE_MARGIN_OF_ERROR,
                minimum_margin_of_error=ABSOLUTE_MARGIN_OF_ERROR,
                confidence_level=CONFIDENCE_LEVEL, 
                batch_size=PARALLEL_BATCH_SIZE,
                log_progress=PRINT_BATCH_PROGRESS
            )
            results_filepath = record_batch_precision_results(results)
            
        elif BATCH_MODE == 'fixed_sample_size':
            results = estimate_stochastic_stats_fixed_size(
                process=simulate_bike_share,
                args=(True, True, RANDOMIZE_STATIONS),
                total_samples=FIXED_SAMPLE_SIZE,
                max_runtime=MAX_RUNTIME,
                batch_size=PARALLEL_BATCH_SIZE,
                confidence_level=CONFIDENCE_LEVEL,
                log_progress=PRINT_BATCH_PROGRESS
            )
            results_filepath = record_batch_fixed_results(results)

    # Print results filepath for frontend
    if USING_APP:
        print(results_filepath, end='')
    
    # Cleanup
    logging.shutdown()
    
if __name__ == "__main__":
    main()