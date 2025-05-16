extends Node2D
## Controls the map of stations, and the markers on it.

const MAP_WIDTH = 1775
const MAP_HEIGHT = 1594

var Marker = preload("res://map/marker/marker.tscn")
@export var markers_container : Node  # markers container
var _markers_list : Array
var _marker_scale = 0.5
const MAX_MARKER_SCALE = 0.75
const MIN_MARKER_SCALE = 0.4

# Parameters for translating from lat/lon to pos
var _center_WGS_coord : Vector2 
var _x_scale : float
var _y_scale : float

@export var camera : Camera2D

# Tracking marker states
var selected_station : int = -1
var _start_station : int = -1
var _end_station : int = -1

# Map paths
var bike_paths : Array
var walk_paths : Array
@export var path_drawer : Node2D
var _visited_stations : Dictionary  # (set) Stations visited by agent
var _results_start_station : int = -1
var _results_end_station : int = -1
var _is_showing_results : bool = false
var _single_run_results_loaded : bool = false

@export var _map_results_overlay : Control
@export var _single_run_results_overlay : Control
@export var _batch_results_overlay : Control

enum marker_sprite_frame {
		BLANK,
		START,
		END,
		START_END,
		VISITED
}

signal station_selected(station : int)


#-------------------------------- Initialization -------------------------------

func _ready():
	_set_coord_transformation()
	_instance_markers()


func _on_tools_external_paths_set():
	_set_coord_transformation()
	_initialize_paths()


func _process(_delta):
	var marker_scale = Vector2.ONE / camera.zoom * 0.5
	marker_scale.x -= 0.1
	marker_scale.x = max(MIN_MARKER_SCALE, marker_scale.x)
	marker_scale.x = min(MAX_MARKER_SCALE, marker_scale.x)
	marker_scale.y = marker_scale.x
	for marker in _markers_list:
		marker.scale = marker.scale.slerp(marker_scale, 0.9)


## Set up biking and walking paths for drawing to map
func _initialize_paths() -> void:
	bike_paths = Tools.load_json_array(Tools.BIKE_PATHS_FILEPATH)
	walk_paths = Tools.load_json_array(Tools.WALK_PATHS_FILEPATH)
	# Convert paths from array of arrays to array of Vector2s
	_convert_paths(bike_paths)
	_convert_paths(walk_paths)
	path_drawer._bike_paths = bike_paths
	path_drawer._walk_paths = walk_paths
	path_drawer.queue_redraw()


## Converts the given lat/lon paths to worldspace vectors, 
## where path is paths[<start_station>][<end_station>]
func _convert_paths(paths : Array) -> void:
	for start_station in range(len(paths)):
		for end_station in range(len(paths)):
			for i in range(len(paths[start_station][end_station])):
				var coords : Array = paths[start_station][end_station][i]
				paths[start_station][end_station][i] = WGS_to_pos(Vector2(coords[0], coords[1]))


## Creates markers for every station and adds them to the map
func _instance_markers() -> void:
	# Get coords from json
	var coords = Tools.load_json_array(Tools.STATION_COORDS_PATH)
	# Iteratively instance markers
	for i in coords.size():
		var coord = coords[i]
		coord = Vector2(coord[0], coord[1])
		var marker = Marker.instantiate()
		# Set up marker
		marker.position = WGS_to_pos(coord)
		marker.scale = Vector2(_marker_scale, _marker_scale)
		marker.set_label(str(i))
		marker._station = i
		# Set z index based on y position so lower markers are in front
		marker.z_index = int(marker.position.y/10 + 300)
		# Connect signals
		marker.marker_button_down.connect(_on_marker_button_down)
		marker.marker_button_up.connect(_on_marker_button_up)
		markers_container.add_child(marker)
		_markers_list.append(marker)


#---------------------------- Position Transform -------------------------------

## Sets up parameters for transforming lat/lon to pos
func _set_coord_transformation() -> void:
	var bottom_left_WGS = Vector2(40.6902 , -74.03786)
	var bottom_left_pos = Vector2(-1775, 1594)
	var top_right_WGS = Vector2(40.74966, -73.95048)
	var top_right_pos = Vector2(1775, -1594)
	var lat_diff_WGS = top_right_WGS.x - bottom_left_WGS.x
	var lon_diff_WGS = top_right_WGS.y - bottom_left_WGS.y
	_center_WGS_coord = Vector2(
		bottom_left_WGS.x + lat_diff_WGS/2, 
		bottom_left_WGS.y + lon_diff_WGS/2
		)
	var x_diff_pos = top_right_pos.x - bottom_left_pos.x
	var y_diff_pos = top_right_pos.y - bottom_left_pos.y
	_x_scale = x_diff_pos / lon_diff_WGS
	_y_scale = y_diff_pos / lat_diff_WGS


## Returns the world position corresponding to the given (lat, lon)
func WGS_to_pos(coord : Vector2) -> Vector2:
	# Use y coordinates (longitude) for x position
	var pos_x = _x_scale * (coord.y - _center_WGS_coord.y)
	var pos_y = _y_scale * (coord.x - _center_WGS_coord.x)
	return Vector2(pos_x, pos_y)


#------------------------------- Adjust Map ------------------------------------

## Sets the station outline to the given station
func _set_selected_station_outline(station):
	# Remove outline of previously selected station
	if selected_station >= 0:
		_markers_list[selected_station].outline.visible = false
	# Add outline to selected station
	selected_station = station
	_markers_list[selected_station].outline.visible = true


## Sets _visited_stations based on the given actions from sim results
func _set_visited_stations(actions : Array) -> void:
	# If showing results, deselect previous start/end result stations
	if _is_showing_results and _results_start_station >= 0:
		_hide_visited_stations()
		#_markers_list[_results_start_station].unset_start_sprite()
		#_markers_list[_results_end_station].unset_end_sprite()
	# Set start/end stations
	_results_start_station = actions[0]['start_station']
	_results_end_station = actions[-1]['end_station']
	print('result stations set')
	print('result_start ', _results_start_station)
	print('result_end ', _results_end_station)
	# Reselect visited stations based on actions
	_visited_stations.clear()
	for action in actions:
		var station : int = action['end_station']
		# If not start/end station, set marker sprite to visited
		if station != _results_start_station and station != _results_end_station:
			_visited_stations[station] = true


## Highlights visited stations, including start/end
func _show_visited_stations():
	print('showing visited stations')
	print('start ', _start_station)
	print('end ', _end_station)
	print('result_start ', _results_start_station)
	print('results_end ', _results_end_station)
	# Set start end markers to results
	_markers_list[_start_station].unset_start_sprite()
	_markers_list[_end_station].unset_end_sprite()
	_markers_list[_results_start_station].set_start_sprite()
	_markers_list[_results_end_station].set_end_sprite()
	# Highlights visited stations on map
	for station in _visited_stations:
		var marker = _markers_list[station]
		marker.set_sprite(marker_sprite_frame.VISITED)


## Hides the visited stations
func _hide_visited_stations():
	print('hiding visited stations')
	print('start ', _start_station)
	print('end ', _end_station)
	print('result_start ', _results_start_station)
	print('results_end ', _results_end_station)
	# De-highlight visited stations on map
	for station in _visited_stations:
		_markers_list[station].set_sprite(marker_sprite_frame.BLANK)
	# Set start end markers to user params
	_markers_list[_results_start_station].unset_start_sprite()
	_markers_list[_results_end_station].unset_end_sprite()
	_markers_list[_start_station].set_start_sprite()
	_markers_list[_end_station].set_end_sprite()


## Sets visited stations and paths based on the given actions
func _set_map_results(actions : Array):
	path_drawer.set_excursion_paths(actions)
	_set_visited_stations(actions)


## Shows currently loaded map results
func _show_map_results():
	# For single run, draw path and show visited stations
	if _single_run_results_loaded and _results_start_station >= 0:
		path_drawer.draw_excursion_paths()
		_show_visited_stations()
		_single_run_results_overlay.visible = true
		_batch_results_overlay.visible = false
	# For batch, hide start/end stations
	else:
		path_drawer.hide_excursion_paths()
		_hide_visited_stations()
		_markers_list[_start_station].unset_start_sprite()
		_markers_list[_end_station].unset_end_sprite()
		_single_run_results_overlay.visible = false
		_batch_results_overlay.visible = true


# Hides map results
func _hide_map_results():
	# For single run, hide path and visited stations
	if _single_run_results_loaded and _results_start_station >= 0:
		path_drawer.hide_excursion_paths()
		_hide_visited_stations()
	# For batch, show start/end stations
	else:
		_markers_list[_start_station].set_start_sprite()
		_markers_list[_end_station].set_end_sprite()
	_single_run_results_overlay.visible = false
	_batch_results_overlay.visible = false


#----------------------------- Signal Responses --------------------------------

func _on_marker_button_down(station):
	camera.input_enabled = false
	_set_selected_station_outline(station)
	station_selected.emit(station)


func _on_marker_button_up():
	camera.input_enabled = true


func _on_station_spinbox_value_changed(value):
	_set_selected_station_outline(value)


func _on_param_controller_station_selected(station):
	_set_selected_station_outline(station)


func _on_start_station_set(station):
	# If showing results, just update start station
	if _is_showing_results:
		_start_station = station
	# Otherwise update markers
	else:
		# Remove previous start station
		if _start_station >= 0:
			_markers_list[_start_station].unset_start_sprite()
		# Set new start station
		_start_station = station
		_markers_list[_start_station].set_start_sprite()


func _on_end_station_set(station):
	# If showing results, just update end station
	if _is_showing_results:
		_end_station = station
	# Otherwise update markers
	else:
		# Remove previous end station
		if _end_station >= 0:
			_markers_list[_end_station].unset_end_sprite()
		# Set new end station
		_end_station = station
		_markers_list[_end_station].set_end_sprite()


func _on_path_results_loaded(actions):
	_single_run_results_loaded = true
	_set_map_results(actions)
	_show_map_results()


func _on_batch_results_loaded():
	_single_run_results_loaded = false
	_show_map_results()


func _on_show_results():
	_show_map_results()
	_is_showing_results = true
	_map_results_overlay.visible = true


func _on_hide_results():
	_hide_map_results()
	_is_showing_results = false
	_map_results_overlay.visible = false
