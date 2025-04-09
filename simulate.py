# Main simulation
import json
from dataclasses import dataclass
import numpy as np
import logging
import time
import os

        
# Unless otherwise indicated:
# - "station" refers to station index
# - "time" refers to simulated time measured in hours

#------------------- USER PARAMETER OVERRIDE --------------------
OVERRIDE_USER_PARAMS = False 
# The following parameters are only used if OVERRIDE_USER_PARAMS is True.
# Otherwise parameters are loaded from file
SEED = None  # Unsigned 32 bit int (if None, one will be generated)
BATCH_SIZE = 1  # Number of simulation replications
START_STATION = 0  # Agent start station index
FINAL_STATION = 0  # Agent final station index
EXCURSION_TIME = 4.0  # Length of excursion in hours
AGENT_INTELLIGENCE = 'smart'  # 'basic', 'smart'
WARM_UP_TIME = 2.0  # The number of hours that the simulation runs before starting the agent
EMPTY_BIAS = 0.0  # Bias towards emptying stations (0-1)
FULL_BIAS = 0.0  # Bias towards filling stations (0-1)

#=============================== STATIC PARAMETERS ===========================================

#-------------------- CONFIGURATION --------------------
START_TIME = 16.0  # Time of day (HH) when the simulation begins

ESTIMATE_FAIL_COUNT_TIME_HORIZON = 1.0  # The amount of time in hours used for
                                        # the fail count estimation sub-simulation
ESTIMATE_FAIL_COUNT_IS_STOCHASTIC = False  # Determines the estimate_fail_count function                                     
ESTIMATE_FAIL_COUNT_REP_COUNT = 20  # Number of simulation replications for stochastic 
                                    # estimation of fail count                                                              
                                        
AGENT_SEARCH_BRANCH_FACTOR = 3  # The number of nearest stations (with rewards if biking) to
                                # search when expanding a node
AGENT_MAX_SEARCH_DEPTH = 4  # The max depth of the agent's search tree
AGENT_MAX_SEARCH_TIME = 2.0  # This is the maximum time from the root node that a node can have.
                           # (Shouldn't be reached, only used to generated future
                           # incentives)
AGENT_WAIT_TIME = 5.001/60  # The length of time in hours the agent waits when no station found
AGENT_WAIT_TIME_LENIENCY = 2/60  # If the wait time ends at most this much time before
                                 # the update, then agent extends wait till update
INCENTIVE_COST = 0.5  # The number of failures that must be mitigated to warrant
                    # incentivizing a station
if not ESTIMATE_FAIL_COUNT_IS_STOCHASTIC:  # Deterministic doesn't work with incentive cost
    INCENTIVE_COST = 0                     # because it uses int estimates
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
    USER_PARAMS_FILEPATH = 'data/user_params.json'

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
BIKE_SYSTEM_PARAMS_FILEPATH = 'data/sim_params.json'

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
#============================================================================================

#----------------------------------------- LOGGING ------------------------------------------
# Levels: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
SINGLE_RUN_LOG_LEVEL = logging.INFO
BATCH_LOG_LEVEL = logging.ERROR
# Time range for debug log (HH)
DEBUG_START_TIME = 18 + 12/60
DEBUG_END_TIME = 18 + 28/60

WRITE_LOG_FILE = True

def generate_log_filepath(seed, log_dir='logs'):
    """ Generates a log filepath in the format data/YYMMDD_HHMM_s<seed>.log """
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

    def update(self, new_station: int, reward_gain: float) -> None:
        """ Updates the agent's state based on the given trip data. """
        # Update station
        self.station = new_station
        # Alternate mode
        self.mode = 'walk' if self.mode == 'bike' else 'bike'
        # Adjust reward
        self.reward += reward_gain    

    def get_action(self, bike_counts: list, incentives: list, current_time: float) -> int:
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
                    print(f'eta: {format_time(return_time)}, node time: {format_time(node.time)}')
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
                        incentive = generate_single_incentive(node.station, old_bike_count)
                        rent_reward = get_reward(incentive, 'rent')
                        incentives_cache[(incentive_time, node.station)] = incentive
                    # Check if end station incentive is already cached
                    if (incentive_time, end_station) in incentives_cache:
                        return_reward = get_reward(incentives_cache[(incentive_time, end_station)], 'return')
                    else:
                        old_bike_count = estimate_bike_count(end_station, root_bike_counts[end_station], -1 * incentive_time_from_root)
                        incentive = generate_single_incentive(end_station, old_bike_count)
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

    def _predict_failure(self, station: int, option: str, bike_count: int):
        """ Returns true if a failure for the given option is expected at the given station 
        within ESTIMATE_FAIL_COUNT_TIME_HORIZON. Returns false if a failure for the 
        alternate option is predicted, or if no failures are predicted.
        Args:
            option (str): {'rent', 'return'} The type of failure to predict
            bike_offset (int): The bike count is offest by this amount
        """
        # Get net rate of bikes being added
        net_rate = RETURN_RATES[station] - RENT_RATES[station]
        # Get new bike count
        bike_difference = net_rate * ESTIMATE_FAIL_COUNT_TIME_HORIZON
        bike_difference = bike_difference
        bike_count += bike_difference
        # Predict rental failure
        if option == 'rent' and bike_count < 0: 
            return True
        # Predict return failure
        if option == 'return' and bike_count > CAPACITIES[station]:
            return True
        # No failures predicted for given option
        return False
           
    def _validate_station(self, station: int, bike_count: int, mode: str) -> bool:
        """ Returns true iff the station is a valid destination. That means going here
        won't fill up or empty the station too much, resulting in failures.
        Args:
            mode (str): {'walk', 'bike'} the mode of travel used to reach
                the given station
        """
        # Predict if going to station would result in failure
        if (
            mode == 'bike'
            and self._predict_failure(station, 'return', bike_count + 1)
            ):
            return False
        elif (
            mode == 'walk'
            and self._predict_failure(station, 'rent', bike_count - 1)
            ):
            return False
        # No failures predicted
        return True


def format_time(time: float) -> str:
    """ Takes a float representing the time of day in hours and returns
    a string in HH:mm format. """
    hours = int(time)
    minutes = int((time - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"


def print_bike_counts(bike_counts: list) -> None:
    """ Prints a list of every station's bike count and capacity. """
    print(f'--- Bike Counts ---')
    for i in range(N):
        print(f'{i}: ({bike_counts[i]}/{CAPACITIES[i]})')
    print('-------------------')


def simulate_batch(batch_size: int) -> dict:
    """ Runs a batch of bike share simulations and returns stats """
    # Set logger level
    logger.setLevel(BATCH_LOG_LEVEL) 
    # Ensure SEED is not preset
    if SEED != None:
        logger.error(f'{Error} SEED should be None for batch simulation.')
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
        print(f'\rRunning simulation batch: {i+1}/{batch_size} complete', end='')
    # Analyze batch data
    print(f'\n------- Batch Complete ({batch_size} runs) --------')
    for key in batch_data:
        logger.error(f'{key}:')
        logger.error(f'\tMean: {np.mean(batch_data[key]):.2f}')
        logger.error(f'\tMedian: {np.median(batch_data[key]):.2f}')
        logger.error(f'\tStd Dev: {np.std(batch_data[key]):.2f}')
        logger.error(f'\tMax: {np.max(batch_data[key]):.2f}')
        logger.error(f'\tMin: {np.min(batch_data[key]):.2f}')


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
    incentives = generate_incentives(bike_counts)
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
        if end_station == NULL_STATION: # and agent.mode == 'walk'
            logger.warning(f'\t{WARNING} No trip found, trying walking')
            #--------------- Try walking ------------------
            # If biking, try one-off walking attempt before waiting
            if agent.mode == 'bike':
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
                    logger.error(f'\nSEED: {seed}')
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
                    incentives = generate_incentives(bike_counts)              
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
            new_incentives = generate_incentives(bike_counts)
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
            reward_str = f'+{reward}'
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


def estimate_fail_count(station: int, bike_count: int, bike_offset: int, rep_count: int=ESTIMATE_FAIL_COUNT_REP_COUNT) -> int:
    if ESTIMATE_FAIL_COUNT_IS_STOCHASTIC:
        return estimate_fail_count_stochastic(station, bike_count, bike_offset, rep_count)
    else:
        return estimate_fail_count_deterministic(station, bike_count, bike_offset)


def estimate_fail_count_stochastic(station: int, bike_count: int, bike_offset: int, rep_count: int=ESTIMATE_FAIL_COUNT_REP_COUNT) -> int:
    """ Estimates the number of failures at the given station in a finite time horizon
    using a sub-simulation. Returns None if given bike_offset puts the bike count out
    of bounds.
    bike_count: number of bikes at station
    bike_offset: offset of number of bikes """
    # Get station parameters
    capacity = CAPACITIES[station]
    agg_rate = RETURN_RATES[station] + RENT_RATES[station]
    # Adjust bike count and return LARGE_INT (like inf) to indicate
    # impossible scenario if bike count is out of bounds
    initial_bike_count = bike_count + bike_offset
    if initial_bike_count < 0 or initial_bike_count > capacity:
        #print(f'>>>{station:3d} | bike_count:{bike_count}, init_bike_count:{initial_bike_count}, capacity:{capacity}')
        return LARGE_INT
    # Get probability that an arrival is a rental
    p_rent = RENT_RATES[station]/agg_rate
    # Repeat sub-simulation and get average of values
    values = []
    for _ in range(rep_count):
        # Set time and bike count
        T = ESTIMATE_FAIL_COUNT_TIME_HORIZON
        t = 0  # current time
        bike_count = initial_bike_count
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
            # Determine arrival option and adjust bike count
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
        values.append(fail_count)
    # Get average fail count
    return float(np.mean(values))


def estimate_fail_count_deterministic(station: int, bike_count: int, bike_offset: int) -> int:
    # Adjust bike count and return LARGE_INT (like inf) to indicate
    # impossible scenario if bike count is out of bounds
    bike_count += bike_offset
    if bike_count < 0 or bike_count > CAPACITIES[station]:
        return LARGE_INT
    # Set time and bike count
    T = ESTIMATE_FAIL_COUNT_TIME_HORIZON
    t = 0
    # Track failures
    rent_fail_count = 0
    return_fail_count = 0
    # Set up clocks
    rates = [RENT_RATES[station], RETURN_RATES[station]]
    rent_clock = 1/rates[0]
    return_clock = 1/rates[1]
    clocks = [rent_clock, return_clock]
    while True:
        # Get event
        event = np.argmin(clocks)
        # Get holding time
        tau = clocks[event]
        # Adjust clocks
        clocks = [clock - tau for clock in clocks]
        clocks[event] = 1/rates[event]
        t += tau
        # Check time horizon
        if t >= T:
            break
        # Update bikes
        if event == 0:  # Rental
            bike_count -= 1
            # Failed rental
            if bike_count < 0:
                rent_fail_count += 1
                bike_count = 0
        elif event == 1:  # Return
            bike_count += 1
            # Failed return
            if bike_count > CAPACITIES[station]:
                return_fail_count += 1
                bike_count = CAPACITIES[station]
    # Simulation complete
    return rent_fail_count + return_fail_count


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


def generate_incentives(bike_counts: list) -> list:
    """ Returns a list of incentives based on the given bike counts """
    incentives = [0] * N
    for station in range(N):
        incentives[station] = generate_single_incentive(station, bike_counts[station])
    return incentives


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
    
    
def main():
    # Run simulation directly for single run
    if BATCH_SIZE == 1:
        # Set logger level
        logger.setLevel(SINGLE_RUN_LOG_LEVEL)
        simulate_bike_share()
    # Otherwise run batch (limits logging)
    else:
        simulate_batch(BATCH_SIZE)
    

if __name__ == "__main__":
    main()