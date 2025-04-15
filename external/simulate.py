# Main simulation
import json
from dataclasses import dataclass
import logging
import time
import os
from queue import PriorityQueue

import numpy as np
from scipy import stats

from tools import estimate_stochastic_mean
        
# Unless otherwise indicated:
# - "station" refers to station index
# - "time" refers to simulated time measured in hours

#------------------- USER PARAMETER OVERRIDE --------------------
OVERRIDE_USER_PARAMS = True  # Must be False for application to work
# The following parameters are only used if OVERRIDE_USER_PARAMS is True.
# Otherwise parameters are loaded from file
SEED = None  # Unsigned 32 bit int (if None, one will be generated)
BATCH_SIZE = 1  # Number of simulation replications
START_STATION = 0  # Agent start station index
FINAL_STATION = 0  # Agent final station index
EXCURSION_TIME = 4.0  # Length of excursion in hours
AGENT_INTELLIGENCE = 'basic'  # 'basic', 'smart'
WARM_UP_TIME = 4.0  # The number of hours that the simulation runs before starting the agent
EMPTY_BIAS = 0.0  # Bias towards emptying stations (0-1)
FULL_BIAS = 0.0  # Bias towards filling stations (0-1)

# Establish data directory
if os.path.exists('external'):
    BASE_PATH = 'external/'
else:
    BASE_PATH = ''

#=============================== STATIC PARAMETERS ===========================================

#-------------------- CONFIGURATION --------------------
START_TIME = 16.0  # Time of day (HH) when the simulation begins                                                     
                                        
AGENT_SEARCH_BRANCH_FACTOR = 6  # The number of nearest stations (with rewards if biking) to
                                # search when expanding a node
AGENT_MAX_SEARCH_DEPTH = 4  # The max depth of the agent's search tree
AGENT_MAX_SEARCH_TIME = 2.0  # This is the maximum time from the root node that a node can have.
                           # (Shouldn't be reached, only used to generated future
                           # incentives)
AGENT_WAIT_TIME = 5.001/60  # The length of time in hours the agent waits when no station found
AGENT_WAIT_TIME_LENIENCY = 2/60  # If the wait time ends at most this much time before
                                 # the update, then agent extends wait till update
AGENT_MAX_WALK_TIME = 5/60  # The maximum time in hours the agent will walk to a station (for basic only)
INCENTIVE_COST = 0.5  # The number of failures that must be mitigated to warrant
                    # incentivizing a station
INCENTIVE_UPDATE_INTERVAL = 0.25  # The period of time in hours between incentives updates

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

#------------------- USER PARAMETERS -------------------
if not OVERRIDE_USER_PARAMS:
    USER_PARAMS_FILEPATH = BASE_PATH + 'data/user_params.json'
    with open(USER_PARAMS_FILEPATH, 'r') as file:
        user_params = json.load(file)

    SEED = user_params['seed']  # Unsigned 32 bit int (if None, one will be generated)
    BATCH_SIZE = user_params['batch_size']  # Number of simulation replications
    START_STATION = user_params['start_station']  # Agent start station index
    FINAL_STATION = user_params['end_station']  # Agent final station index
    EXCURSION_TIME = user_params['excursion_time']  # Length of excursion in hours
    AGENT_INTELLIGENCE = user_params['agent_mode']  # 'basic', 'smart'
    WARM_UP_TIME = user_params['warmup_time']  # The number of hours that the simulation runs before starting the agent
    EMPTY_BIAS = user_params['empty_bias']  # Bias towards emptying stations (0-1)
    FULL_BIAS = user_params['full_bias']  # Bias towards filling stations (0-1)

#--------------- BIKE SYSTEM PARAMETERS ----------------

BIKE_SYSTEM_PARAMS_FILEPATH = BASE_PATH + 'data/sim_params.json'

# Load bike system parameters for unpacking
with open(BIKE_SYSTEM_PARAMS_FILEPATH, 'r') as file:
    params = json.load(file)

# Number of stations
N = len(params['capacities'])

# Lists indexed by station (length N)
#RENT_RATES = [x * 10 for x in params['rent_rates'] ] # floats
RENT_RATES = params['rent_rates']
RETURN_RATES = params['return_rates']  # floats
INITIAL_BIKE_COUNTS = params['initial_bike_counts']  # ints
CAPACITIES = params['capacities']  # ints

# 2D lists indexed by station ([N x N] float matrix)
DEST_PROBS = params['dest_probs']  # Destination probability matrix
BIKE_TIMES = params['bike_times']
WALK_TIMES = params['walk_times']

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
NEAR_BIKE_STATIONS = get_near_stations_list(params['bike_times'])
NEAR_WALK_STATIONS = get_near_stations_list(params['walk_times'])

# Load incentives from file
INCENTIVES_FILEPATH = BASE_PATH + 'data/incentives.json'
with open(INCENTIVES_FILEPATH, 'r') as file:
    # INCENTIVES[<station>][<bike_count>]
    INCENTIVES = json.load(file)

def adjust_incentives():
    """ Subtract incentive cost from incentives """
    pass
    # ToDo
    
    
    
    
    
#============================================================================================

#----------------------------------------- LOGGING ------------------------------------------
# Levels: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
SINGLE_RUN_LOG_LEVEL = logging.INFO
BATCH_LOG_LEVEL = logging.ERROR
# Time range for debug log (HH)
DEBUG_START_TIME = 18 + 12/60
DEBUG_END_TIME = 18 + 28/60

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
            return self.get_smart_action(bike_counts, incentives, current_time)
        elif AGENT_INTELLIGENCE == 'basic':
            return self.get_basic_action(bike_counts, incentives, current_time)

    def get_basic_action(self, bike_counts: list, incentives: list, current_time: float) -> int:
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

    def get_smart_action(self, bike_counts: list, incentives: list, current_time: float) -> int:
        """ Returns the next station the agent will travel to using a search tree. """
        # Estimate and cache future incentives for comparing nodes
        incentives_cache = dict() # {(<update_time>, <station>): <incentive>}     
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
            return_time = node.time + WALK_TIMES[node.station][self.final_station]
            #---------------- Max time exceeded ----------------
            if return_time > self.end_time:
                # End trip if root node is overtime
                if node.prev == None:
                    logger.error(f'eta: {format_time(return_time)}, node time: {format_time(node.time)}')
                    return END_TRIP
                # Otherwise evaluate previous node
                if is_best(node.prev):
                    best_node = node.prev
            #---------------- Max depth reached ---------------
            elif node.depth == AGENT_MAX_SEARCH_DEPTH:
                # Evaluate current node
                if is_best(node):
                    best_node = node
            #----------------- Expand node --------------------
            else:
                new_nodes = self._expand_node(node, bike_counts, incentives, root_time, incentives_cache)
                search_queue += new_nodes
                # Log tree expansion
                if logger.level <= logging.DEBUG and (DEBUG_START_TIME < root_time < DEBUG_END_TIME):
                    logger.debug(f'{node.station}, time: {format_time(node.time)}, reward: {node.reward}, {node.mode} to {[n.station for n in new_nodes]}')
                #--------------------- No trips available  --------------------
                if not new_nodes:
                    #------------- Evaluate current node --------------
                    # Ensure there was at least one bike trip in this branch
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

    def _expand_node(self, node: Node, root_bike_counts: list, root_incentives: list, root_time: float, incentives_cache: dict) -> list:
        """ Expand search node using nearest stations. Returns list of nodes. 
        Args:
            root_bike_counts (list): the bike_counts at the time of the root node 
        """
        # Get nearest stations based on transport mode
        if node.mode == 'bike':
            near_stations = NEAR_BIKE_STATIONS[node.station]
        elif node.mode == 'walk':
            near_stations = NEAR_WALK_STATIONS[node.station]
        # Get new node for each station
        branch_factor = AGENT_SEARCH_BRANCH_FACTOR  # Number of stations to search
        new_nodes = []
        for end_station in near_stations:
            #--------------------- Estimate Reward ---------------------
            # Only biking yields rewards
            if node.mode == 'bike':
                # Get last incentive update time
                incentive_time = np.floor(node.time / INCENTIVE_UPDATE_INTERVAL) * INCENTIVE_UPDATE_INTERVAL
                # If last update was before root time, use current incentives
                if incentive_time <= root_time:
                    rent_reward = get_reward(root_incentives[node.station], 'rent')
                    return_reward = get_reward(root_incentives[end_station], 'return')
                    reward = rent_reward + return_reward
                # Otherwise, estimate incentives from last update
                else:  # incentive_time > root_time
                    incentive_time_from_root = incentive_time - root_time
                    # Check if start station incentive is already cached
                    if (incentive_time, node.station) in incentives_cache:
                        rent_reward = get_reward(incentives_cache[(incentive_time, node.station)], 'rent')
                    else:
                        old_bike_count = estimate_bike_count(node.station, root_bike_counts[node.station], -1 * incentive_time_from_root)
                        incentive = INCENTIVES[node.station][old_bike_count]
                        rent_reward = get_reward(incentive, 'rent')
                        incentives_cache[(incentive_time, node.station)] = incentive
                    # Check if end station incentive is already cached
                    if (incentive_time, end_station) in incentives_cache:
                        return_reward = get_reward(incentives_cache[(incentive_time, end_station)], 'return')
                    else:
                        old_bike_count = estimate_bike_count(end_station, root_bike_counts[end_station], -1 * incentive_time_from_root)
                        incentive = INCENTIVES[end_station][old_bike_count]
                        return_reward = get_reward(incentive, 'return')
                        incentives_cache[(incentive_time, end_station)] = incentive                        
                    reward = rent_reward + return_reward
                # Go to next station if no reward for biking
                if reward == 0:
                    continue
            # If walking, there is no reward
            else:  # node.mode == 'walk'
                reward = 0
            #-------------------- Validate station ---------------------
            # Get travel time    
            travel_times = BIKE_TIMES if node.mode =='bike' else WALK_TIMES
            trip_duration = travel_times[node.station][end_station]
            end_time = node.time + trip_duration
            # Estimate bike count at arrival time
            end_bike_count = estimate_bike_count(end_station, root_bike_counts[end_station], end_time - root_time)
            # If station is not valid, go to next station
            if not self._validate_station(end_station, end_bike_count, node.mode):
                continue
            #------------------ Instantiate new node -------------------        
            # Alternate transport mode
            mode = 'walk' if node.mode == 'bike' else 'bike'
            # Set action if previous node is root, otherwise use previous action
            action = end_station if node.action == NULL_STATION else node.action
            # Create node and add to expanded list
            new_node = Node(action, end_station, mode, node.root_mode, end_time, node.reward + reward, node.depth+1, node)
            new_nodes.append(new_node)
            # Loop until branching factor is met
            if len(new_nodes) == branch_factor:
                break
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
    a string in HH:mm format. 
    """
    hours = int(time)
    minutes = int((time - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"


def log_bike_counts(bike_counts: list) -> None:
    """ Prints a list of every station's bike count and capacity. """
    logger.info(f'--- Bike Counts ---')
    for i in range(N):
        logger.info(f'{i}: ({bike_counts[i]}/{CAPACITIES[i]})')
    logger.info('-------------------')


def simulate_batch(batch_size: int) -> dict:
    """ Runs a batch of bike share simulations and returns stats """
    # Set logger level
    logger.setLevel(BATCH_LOG_LEVEL) 
    # Ensure SEED is not preset
    if SEED != None:
        logger.error(f'{ERROR} SEED should be None for batch simulation.')
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
    
    
def log_data_stats(data: dict) -> None:
    for key in data:
        logger.error(f'{key}:')
        logger.error(f'\tMean: {np.mean(data[key]):.2f}')
        logger.error(f'\tMedian: {np.median(data[key]):.2f}')
        logger.error(f'\tStd Dev: {np.std(data[key]):.2f}')
        logger.error(f'\tMax: {np.max(data[key]):.2f}')
        logger.error(f'\tMin: {np.min(data[key]):.2f}')


def simulate_bike_share() -> dict:
    """ Simulates a single excursion of an agent rebalancing bikes in a bike share system,
    and returns a dictionary of stats including rewards earned and trips taken. """
    # Track simulation time
    real_start_time = time.perf_counter()
    #--------------- Set seed ---------------
    if SEED is None:
        seed = np.random.randint(0, 2**32, dtype=np.uint32)  # Generate unsigned 32 bit int
        logger.info(f"Generated random seed: {seed}")
    else:
        seed = SEED
        logger.info(f"Using fixed seed: {SEED}")
    np.random.seed(seed)
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
    # Report simulation data
    logger.info(f"> {agent.station} ({format_time(current_time)})")
    logger.info(f'\n---------- Excursion Complete ----------')
    logger.info(f'End time: {format_time(current_time)}')
    logger.info(f'Final station: {agent.station}')
    logger.info(f'Bike count: {bike_trip_count}')
    logger.info(f'Walk count: {walk_trip_count}')
    logger.info(f'Wait count: {wait_count}')
    logger.info(f'Total reward: {agent.reward}')
    logger.info(f'Seed: {seed}')
    logger.info('')
    # Return stats
    data = dict()
    data['excursion_time'] = current_time - START_TIME
    data['real_time_duration'] = real_time_duration
    data['bike_count'] = bike_trip_count
    data['walk_count'] = walk_trip_count
    data['wait_count'] = wait_count
    data['reward'] = agent.reward
    return data


def generate_bike_counts(bike_counts: list, elapsed_time: float) -> list:
    """ Returns an updated list of bike counts for each station after the elapsed time. """
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


def estimate_bike_count(station: int, bike_count: int, time_difference: float) -> int:
    """ Given the current bike count of a station, returns an estimate of the 
    bike count after the time difference (or before, if time_difference < 0)"""
    # Bike count is unchanged for zero time difference
    if time_difference == 0:
        return bike_count
    # Estimate change in bike count during elapsed time
    net_rate = RETURN_RATES[station] - RENT_RATES[station]
    bike_difference = int(round(time_difference * net_rate))
    # Subtract difference to get previous bike count
    new_bike_count = bike_count + bike_difference
    # Limit by capacity
    new_bike_count = max(0, min(CAPACITIES[station], new_bike_count))
    return new_bike_count


def get_reward(incentive: float, option: str) -> int:
    """ Returns the reward for returning or renting a bike to a station
    with the given incentive. 
    Args:
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
    reward = min(1, reward)
    return reward


def log_incentivized_stations(incentives: list) -> None:
    """ Prints incentivized stations categorized by incentive option (rentals/returns) """
    rent_stations = []
    return_stations = []
    for i, incentive in enumerate(incentives):
        if incentive > 0:
            return_stations.append(i)
        elif incentive < 0:
            rent_stations.append(i)
    logger.info(f'Rent:    {rent_stations}\nReturn:  {return_stations}')
    

def run_sim_and_get_reward() -> float:
    logger.setLevel(BATCH_LOG_LEVEL)
    data = simulate_bike_share()
    return data['reward']
    
def main():    
    print()
    logger.setLevel(BATCH_LOG_LEVEL)
    estimate_stochastic_mean(
        run_sim_and_get_reward, 
        args=(), 
        margin_of_error=0.01, 
        confidence_level=0.999, 
        batch_size=12,
        log_progress=True
    )
    return
    
    # Run simulation directly for single run
    if BATCH_SIZE == 1:
        # Set logger level
        logger.setLevel(SINGLE_RUN_LOG_LEVEL)
        simulate_bike_share()
    # Otherwise run batch (limits logging)
    else:
        simulate_batch(BATCH_SIZE)
    
    file_handler.close()
    logging.shutdown()
    

if __name__ == "__main__":
    main()