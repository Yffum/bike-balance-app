extends Node
## Globally accessible tools

var MAX_LOG_COUNT = 20  # Max number of log files before pruning
var MAX_RESULTS_COUNT = 20 # Max number of results jsons before pruning

#------------------ Colors --------------------
var BIKE_PATH_COLOR = Color(1, 0, 0)
var WALK_PATH_COLOR = Color(0, 0, 1)

#----------------- Internal Paths ------------------
var INTERNAL_DATA_PATH = ProjectSettings.globalize_path('res://data')
var DEFAULT_USER_PARAMS_PATH = INTERNAL_DATA_PATH.path_join('default_user_params.json')
var STATION_COORDS_PATH = INTERNAL_DATA_PATH.path_join('station_coords.json')
var STATION_NAMES_PATH = INTERNAL_DATA_PATH.path_join('station_names.json')
var STATION_IDS_PATH = INTERNAL_DATA_PATH.path_join('station_ids.json')
var BIKE_PATHS_FILEPATH = INTERNAL_DATA_PATH.path_join('bike_paths.json')
var WALK_PATHS_FILEPATH = INTERNAL_DATA_PATH.path_join('walk_paths.json')

#----------------- External Paths ------------------
var EXTERNAL_DIR : String
var EXTERNAL_DATA_PATH : String
var LOGS_PATH : String
var RESULTS_PATH : String
var USER_PARAMS_PATH : String
var SIM_PARAMS_PATH : String
var INCENTIVES_PATH : String
var SIM_SCRIPT_PATH : String

signal external_paths_set

func _ready():
	_set_external_paths()
	external_paths_set.emit()

## Sets global filepaths. Uses internal directory if in editor; uses
## executable base directory otherwise.
func _set_external_paths() -> void:
	# If in editor, use resource paths
	if OS.has_feature('editor'):
		EXTERNAL_DIR = ProjectSettings.globalize_path('res://external')
	# If in exported executable, use base folder
	else:
		EXTERNAL_DIR = OS.get_executable_path().get_base_dir()
		
	EXTERNAL_DATA_PATH = EXTERNAL_DIR.path_join('data')
	LOGS_PATH = EXTERNAL_DIR.path_join('logs')
	RESULTS_PATH = EXTERNAL_DIR.path_join('results')
	USER_PARAMS_PATH = EXTERNAL_DATA_PATH.path_join('user_params.json')
	SIM_PARAMS_PATH = EXTERNAL_DATA_PATH.path_join('sim_params.json')
	INCENTIVES_PATH = EXTERNAL_DATA_PATH.path_join('incentives.json')
	SIM_SCRIPT_PATH = EXTERNAL_DIR.path_join('simulate.py')
	
#----------------------- Global Tool Functions ---------------------------

## Returns an array of the JSON file at the given path
func load_json_array(path: String) -> Array:
	# Get data
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("Failed to open JSON file: %s" % path)
		return []
	# Convert to array
	var content := file.get_as_text()
	var result: Variant = JSON.parse_string(content)
	if result == null:
		push_error("Failed to parse JSON.")
		return []
	# Ensure data is array
	if result is Array:
		return result as Array
	else:
		push_error("JSON data is not a Array.")
		return []


## Returns a Dictionary of the JSON file at the given path
func load_json_dict(path: String) -> Dictionary:
	# Get data
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("Failed to open JSON file: %s" % path)
		return {}
	# Convert to dict
	var content := file.get_as_text()
	var result: Variant = JSON.parse_string(content)
	if result == null:
		push_error("Failed to parse JSON.")
		return {}
	# Ensure data is dict
	if result is Dictionary:
		return result as Dictionary
	else:
		push_error("JSON data is not a Dictionary.")
		return {}


# Returns a Dictionary or Array of the JSON file at the given path
func load_json(path: String) -> Variant:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error('Failed to open JSON file: %s' % path)
		return null
	var content := file.get_as_text()
	return JSON.parse_string(content)


## Saves the given Dictionary or Array to a JSON file at the given path
func save_json(path: String, data: Variant) -> void:
	# Open the file for writing
	var file = FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		push_error("Failed to open file for writing: %s" % path)
		return
	# Convert the data to a JSON string 
	var json_string: String
	if data is Dictionary or data is Array:
		json_string = JSON.stringify(data, '  ')
	else:
		push_error("Unsupported data type: %s" % data)
		return
	# Write the JSON string to the file
	file.store_string(json_string)
	file.close()
	print("File saved at: %s" % path)
