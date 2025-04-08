extends Camera2D

const MIN_ZOOM = 0.25
const INITIAL_ZOOM = 0.387431
const INITIAL_POSITION = Vector2(-1775.0, -1316.514)

@export var zoom_speed : float = 10
@export var zoom_factor : float = 0.1
@export var map_sprite : Sprite2D # Drag your map Sprite2D node here in the editor

var zoom_target : Vector2
var drag_start_mouse_pos = Vector2.ZERO
var drag_start_camera_pos = Vector2.ZERO
var is_dragging = false
var input_enabled = false

var map_size = Vector2.ZERO

func _ready():
	zoom = Vector2(INITIAL_ZOOM, INITIAL_ZOOM)
	position = INITIAL_POSITION
	
	zoom_target = zoom
	map_size = map_sprite.texture.get_size() * map_sprite.scale


func _process(delta):
	if input_enabled:
		handle_zoom(delta)
		handle_pan()
	elif is_dragging:
		handle_pan()
	clamp_camera_position()
	
	print(zoom)
	print(position)

func handle_zoom(delta: float):
	if Input.is_action_just_pressed('camera_zoom_in'):
		zoom_target *= 1 + zoom_factor
	elif Input.is_action_just_pressed('camera_zoom_out'):
		zoom_target *= 1 - zoom_factor
	zoom_target.x = max(MIN_ZOOM, zoom_target.x)
	zoom_target.y = zoom_target.x
	zoom = zoom.lerp(zoom_target, zoom_speed * delta)

func handle_pan():
	if not is_dragging and Input.is_action_just_pressed('camera_pan'):
		drag_start_mouse_pos = get_viewport().get_mouse_position()
		drag_start_camera_pos = position
		is_dragging = true
	if is_dragging and Input.is_action_just_released('camera_pan'):
		is_dragging = false
	if is_dragging:
		var move_vector = get_viewport().get_mouse_position() - drag_start_mouse_pos
		position = position.lerp(drag_start_camera_pos - move_vector / zoom.x, 1)

func clamp_camera_position():
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
		var centered_pos = Vector2.ZERO - viewport_size / (2 * zoom)
		position = position.lerp(centered_pos, 1) # lower weight to smooth transition
	# Otherwise update position
	else:
		position = new_pos

func _on_map_container_mouse_entered():
	input_enabled = true

func _on_map_container_mouse_exited():
	input_enabled = false
