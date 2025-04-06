extends Node

const MAP_WIDTH = 1775
const MAP_HEIGHT = 1594

var Marker = preload("res://map/marker/marker.tscn")
var markers : Node  # markers container
var _marker_scale = 0.5

# Parameters for translating from lat/lon to pos
var _center_WGS_coord : Vector2 
var _x_scale : float
var _y_scale : float

func _ready():
	_set_coord_transformation()
	
	markers = $Markers
	_inst_markers()


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func _inst_markers():
	# Get coords from json
	var coords = Tools.load_json_array('res://data/station_coords.json')
	# Iteratively instance markers
	for coord in coords:
		coord = Vector2(coord[0], coord[1])
		var marker = Marker.instantiate()
		marker.position = WGS_to_pos(coord)
		marker.scale = Vector2(_marker_scale, _marker_scale)
		markers.add_child(marker)

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
