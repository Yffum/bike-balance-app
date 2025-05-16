extends Camera2D
## The camera for the station map

const MIN_ZOOM = 0.25
const MAX_ZOOM = 5
const INITIAL_ZOOM = 0.387431
const INITIAL_POSITION = Vector2(-1775.0, -1316.514)

@export var zoom_speed : float = 10
## Zoom factor for mouse scrolling
@export var zoom_factor : float = 0.1
## Zoom factor for buttons
@export var zoom_buttons_factor : float = 0.5
@export var map_sprite : Sprite2D

var zoom_target : Vector2
var drag_start_mouse_pos = Vector2.ZERO
var drag_start_camera_pos = Vector2.ZERO
var is_dragging = false
var input_enabled = false
var map_is_smaller_than_viewport : bool

var map_size : Vector2


func _ready():
	# Set up zoom and position
	zoom = Vector2(INITIAL_ZOOM, INITIAL_ZOOM)
	position = INITIAL_POSITION
	# Get zoom target and map size
	zoom_target = zoom
	map_size = map_sprite.texture.get_size() * map_sprite.scale
	# Clamp position
	map_is_smaller_than_viewport = clamp_camera_position()


func _process(delta):
	# Zoom camera
	handle_zoom(delta)
	# Pan camera
	if not map_is_smaller_than_viewport and input_enabled:
		handle_pan()
	elif is_dragging:
		if not map_is_smaller_than_viewport:
			handle_pan()
	# If zoomed out beyond map borders, clamp position and zoom to center
	map_is_smaller_than_viewport = clamp_camera_position()
	if map_is_smaller_than_viewport:
		is_dragging = false
	

## Zooms the camera based on input
func handle_zoom(delta: float) -> void:
	# Adjust target zoom based on input
	if input_enabled:
		if Input.is_action_just_pressed('camera_zoom_in'):
			zoom_target *= 1 + zoom_factor
		elif Input.is_action_just_pressed('camera_zoom_out'):
			zoom_target *= 1 - zoom_factor
	zoom_target.x = max(MIN_ZOOM, zoom_target.x)
	zoom_target.x = min(MAX_ZOOM, zoom_target.x)
	zoom_target.y = zoom_target.x
	var new_zoom = zoom.lerp(zoom_target, zoom_speed * delta)
	
	# Get ratio from lerp to determine position adjustment
	var zoom_ratio = (new_zoom.x / zoom.x) - 1
	zoom = new_zoom
	
	# Zoom at mouse position
	if not map_is_smaller_than_viewport and input_enabled:
		position += zoom_ratio * get_viewport().get_mouse_position() / zoom
	# Zoom at center
	else:
		position += zoom_ratio * get_viewport().size / 2 / zoom


# Pans the camera based on input
func handle_pan() -> void:
	# Begin pan based on input
	if not is_dragging and Input.is_action_just_pressed('camera_pan'):
		drag_start_mouse_pos = get_viewport().get_mouse_position()
		drag_start_camera_pos = position
		is_dragging = true
	# Stop pan based on input
	if is_dragging and Input.is_action_just_released('camera_pan'):
		is_dragging = false
	# Pan camera
	if is_dragging:
		var move_vector = get_viewport().get_mouse_position() - drag_start_mouse_pos
		position = position.lerp(drag_start_camera_pos - move_vector / zoom.x, 1)


## If camera is zoomed out beyond the borders of the map, centers camera
## and returns true. Otherwise, returns false.
func clamp_camera_position() -> bool:
	var viewport_size = Vector2(get_parent().size)
	var min_pos = map_size * -0.5
	var max_pos = (map_size * 0.5) - viewport_size / zoom
	var new_pos : Vector2
	# Clamp new position to viewport bounds
	new_pos.x = clamp(position.x, min_pos.x, max_pos.x)
	new_pos.y = clamp(position.y, min_pos.y, max_pos.y)
	# Move camera to center if map is too zoomed out to be within bounds
	if (
		new_pos.x > max_pos.x
		or new_pos.x < min_pos.x
		or new_pos.y > max_pos.y
		or new_pos.y < min_pos.y
		):
		# Center camera slowly
		var centered_pos = Vector2.ZERO - viewport_size / (2 * zoom)
		position = position.lerp(centered_pos, 0.05) # lower weight to smooth transition
		return true
	# Camera is in bounds already
	else:
		# Limit moving camera out of bounds
		position = new_pos
		return false


#----------------------- Signal Responses ------------------------------

func _on_map_container_mouse_entered():
	input_enabled = true


func _on_map_container_mouse_exited():
	input_enabled = false


func _on_zoom_in_button_pressed():
	zoom_target *= 1 + zoom_buttons_factor


func _on_zoom_out_button_pressed():
	zoom_target *= 1 - zoom_buttons_factor
