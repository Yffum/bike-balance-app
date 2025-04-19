extends Node2D

var station : int

signal marker_button_down
signal marker_button_up

@export var content : Node2D  # for scaling
@export var fill : Node2D
@export var label : Label
@export var outline : Sprite2D
@export var button : BaseButton

func set_label(text : String):
	label.text = text

func _on_button_down():
	outline.visible = true
	marker_button_down.emit()
	print(station)

func _on_button_up():
	marker_button_up.emit()
	if not button.is_hovered():
		content.scale /= 1.5
	
func _on_mouse_entered():
	move_to_front()
	if not button.is_pressed():
		content.scale *= 1.5

func _on_mouse_exited():
	if not button.is_pressed():
		content.scale /= 1.5
