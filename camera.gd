extends Camera2D

@export var zoom_speed : float = 10;
@export var zoom_factor : float = 0.1;

var zoom_target : Vector2
var drag_start_mouse_pos = Vector2.ZERO
var drag_start_camera_pos = Vector2.ZERO
var is_dragging = false

func _ready():
	zoom_target = zoom
	
func _process(delta):
	handle_zoom(delta)
	handle_pan()
	
func handle_zoom(delta: float):
	if Input.is_action_just_pressed('camera_zoom_in'):
		zoom_target *= 1 + zoom_factor
	elif Input.is_action_just_pressed('camera_zoom_out'):
		zoom_target *= 1 - zoom_factor
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
		position = position.lerp(drag_start_camera_pos - move_vector * 1/zoom.x, 1)
