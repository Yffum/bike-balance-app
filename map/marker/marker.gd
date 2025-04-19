extends Node2D

var station : int

signal marker_button_down
signal marker_button_up

@export var fill : Node2D
@export var label : Label

func _on_button_down():
	marker_button_down.emit()
	print(station)

func _on_button_up():
	marker_button_up.emit()
	
func _on_mouse_entered():
	self.scale *= 1.5

func _on_mouse_exited():
	self.scale /= 1.5
