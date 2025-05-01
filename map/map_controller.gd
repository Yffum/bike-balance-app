extends Node2D

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
var start_station : int = -1
var end_station : int = -1

# Map paths
var bike_paths : Array
var walk_paths : Array
@export var path_drawer : Node2D
var _visited_stations : Dictionary  # (set) Stations visited by agent
var _results_start_station : int = -1
var _results_end_station : int = -1


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


func _process(delta):
	var marker_scale = Vector2.ONE / camera.zoom * 0.5
	marker_scale.x -= 0.1
	marker_scale.x = max(MIN_MARKER_SCALE, marker_scale.x)
	marker_scale.x = min(MAX_MARKER_SCALE, marker_scale.x)
	marker_scale.y = marker_scale.x
	for marker in _markers_list:
		marker.scale = marker.scale.slerp(marker_scale, 0.9)


func _initialize_paths():
	bike_paths = Tools.load_json_array(Tools.BIKE_PATHS_FILEPATH)
	# Convert paths from array of arrays to array of Vector2s
	_convert_paths(bike_paths)
	path_drawer._bike_paths = bike_paths
	
	### TODO
	path_drawer._walk_paths = bike_paths
	
	path_drawer.queue_redraw()


## Converts the given lat/lon paths to worldspace vectors, 
## where path is paths[<start_station>][<end_station>]
func _convert_paths(paths : Array) -> void:
	for start_station in range(len(paths)):
		for end_station in range(len(paths)):
			for i in range(len(paths[start_station][end_station])):
				var coords : Array = paths[start_station][end_station][i]
				paths[start_station][end_station][i] = WGS_to_pos(Vector2(coords[0], coords[1]))


func _instance_markers():
	# Get coords from json
	var coords = Tools.load_json_array(Tools.STATION_COORDS_PATH)
	# Iteratively instance markers
	for i in coords.size():
		var coord = coords[i]
		coord = Vector2(coord[0], coord[1])
		var marker = Marker.instantiate()
		# Set up marker
		print('marker', WGS_to_pos(coord))
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

func _set_selected_station_outline(station):
	# Remove outline of previously selected station
	if selected_station >= 0:
		_markers_list[selected_station].outline.visible = false
	# Add outline to selected station
	selected_station = station
	_markers_list[selected_station].outline.visible = true


## Sets the given marker sprite to start station
func _set_start_station(station):
	_markers_list[station]


func _set_visited_stations(actions : Array) -> void:
	# Set start/end stations
	_results_start_station = actions[0]['start_station']
	_results_end_station = actions[-1]['end_station']
	# Reselect visited stations based on actions
	_visited_stations.clear()
	for action in actions:
		var station : int = action['end_station']
		# If not start/end station, set marker sprite to visited
		if station != _results_start_station and station != _results_end_station:
			_visited_stations[station] = true


## Highlights visited stations, including start/end
func _show_visited_stations():
	# Set start end markers to results
	_markers_list[start_station].unset_start_sprite()
	_markers_list[end_station].unset_end_sprite()
	_markers_list[_results_start_station].set_start_sprite()
	_markers_list[_results_end_station].set_end_sprite()
	# Highlights visited stations on map
	for station in _visited_stations:
		var marker = _markers_list[station]
		marker.set_sprite(marker_sprite_frame.VISITED)


func _hide_visited_stations():
	# De-highlight visited stations on map
	for station in _visited_stations:
		_markers_list[station].set_sprite(marker_sprite_frame.BLANK)
	# Set start end markers to user params
	_markers_list[_results_start_station].unset_start_sprite()
	_markers_list[_results_end_station].unset_end_sprite()
	_markers_list[start_station].set_start_sprite()
	_markers_list[end_station].set_end_sprite()


func _set_map_results(actions : Array):
	path_drawer.set_excursion_paths(actions)
	_set_visited_stations(actions)


func _show_map_results():
	path_drawer.draw_excursion_paths()
	_show_visited_stations()


func _hide_map_results():
	path_drawer.hide_excursion_paths()
	_hide_visited_stations()


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
	# Remove previous start station
	if start_station >= 0 and start_station != station:
		var prev_start_marker = _markers_list[start_station]
		if start_station == end_station:
			prev_start_marker.set_sprite(marker_sprite_frame.END)
		else:
			prev_start_marker.set_sprite(marker_sprite_frame.BLANK)
			prev_start_marker.label.visible = true
	# Set new start station
	start_station = station
	var frame : int = marker_sprite_frame.START
	if end_station == start_station:
		frame = marker_sprite_frame.START_END
	var new_start_marker = _markers_list[start_station]
	new_start_marker.set_sprite(frame)
	new_start_marker.label.visible = false


func _on_end_station_set(station):
	# Remove previous end station
	if end_station >= 0 and end_station != station:
		var prev_end_marker = _markers_list[end_station]
		if start_station == end_station:
			prev_end_marker.set_sprite(marker_sprite_frame.START)
		else:
			prev_end_marker.set_sprite(marker_sprite_frame.BLANK)
			prev_end_marker.label.visible = true
	# Set new end station
	end_station = station
	var frame : int = marker_sprite_frame.END
	if end_station == start_station:
		frame = marker_sprite_frame.START_END
	var new_end_marker = _markers_list[end_station]
	new_end_marker.set_sprite(frame)
	new_end_marker.label.visible = false


func _on_path_results_loaded(actions):
	_set_map_results(actions)


func _on_show_results():
	if _results_start_station >= 0:
		_show_map_results()


func _on_hide_results():
	if _results_start_station >= 0:
		_hide_map_results()
