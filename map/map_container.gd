extends SubViewportContainer



func _ready():
	# Ensure mouse position is tracked at launch
	if get_rect().has_point(get_local_mouse_position()):
		emit_signal('mouse_entered')
	else:
		emit_signal('mouse_exited')
