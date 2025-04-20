extends Node

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

var selected_station : int = -1
signal station_selected(station : int)

func _ready():
	_set_coord_transformation()
	_instance_markers()
	
func _process(delta):
	var marker_scale = Vector2.ONE / camera.zoom * 0.5
	marker_scale.x -= 0.1
	marker_scale.x = max(MIN_MARKER_SCALE, marker_scale.x)
	marker_scale.x = min(MAX_MARKER_SCALE, marker_scale.x)
	marker_scale.y = marker_scale.x
	for marker in _markers_list:
		marker.scale = marker.scale.slerp(marker_scale, 0.9)
	
func _instance_markers():
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
		marker.station = i
		# Connect signals
		marker.marker_button_down.connect(_on_marker_button_down)
		marker.marker_button_up.connect(_on_marker_button_up)
		markers_container.add_child(marker)
		_markers_list.append(marker)

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

func _on_marker_button_down(station):
	camera.input_enabled = false
	_set_selected_station_outline(station)
	station_selected.emit(station)
	
func _on_marker_button_up():
	camera.input_enabled = true
	
func _on_station_spinbox_value_changed(value):
	_set_selected_station_outline(value)
	
func _set_selected_station_outline(station):
	# Remove outline of previously selected station
	if selected_station >= 0:
		_markers_list[selected_station].outline.visible = false
	# Add outline to selected station
	selected_station = station
	_markers_list[selected_station].outline.visible = true

func _on_param_controller_station_selected(station):
	_set_selected_station_outline(station)
