extends Node2D

var station : int

signal marker_button_down(station : int)
signal marker_button_up

const ZOOM_ON_HOVER_SCALE = 1.15

@export var content : Node2D  # for scaling
@export var fill : Node2D
@export var label : Label
@export var outline : Sprite2D
@export var button : BaseButton
@export var anim_sprite : AnimatedSprite2D



func set_label(text : String):
	label.text = text
	
func set_sprite(frame : int):
	anim_sprite.frame = frame

func _on_button_down():
	marker_button_down.emit(station)
	
func _on_button_up():
	marker_button_up.emit()
	if not button.is_hovered():
		content.scale /= ZOOM_ON_HOVER_SCALE
	
func _on_mouse_entered():
	move_to_front()
	if not button.is_pressed():
		content.scale *= ZOOM_ON_HOVER_SCALE

func _on_mouse_exited():
	if not button.is_pressed():
		content.scale /= ZOOM_ON_HOVER_SCALE
