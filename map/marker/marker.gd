extends Node2D

var _station : int

signal marker_button_down(station : int)
signal marker_button_up

const ZOOM_ON_HOVER_SCALE = 1.15

@export var content : Node2D  # for scaling
@export var fill : Node2D
@export var label : Label
@export var outline : Sprite2D
@export var button : BaseButton
@export var anim_sprite : AnimatedSprite2D

enum _frames {
		BLANK,
		START,
		END,
		START_END,
		VISITED
}

## Sprite frames that have textures (so the label should be hidden)
var _textured_frames : Array = [
	_frames.START,
	_frames.END,
	_frames.START_END,
]


## Sets the marker label
func set_label(text : String):
	label.text = text


## Sets sprite to given frame and adjusts label appropriately
func set_sprite(frame : int):
	# Set sprite
	anim_sprite.frame = frame
	# Hide label if sprite is textured
	if frame in _textured_frames:
		set_label('')
	else:
		set_label(str(_station))
	# Use dark text for VISITED
	if frame == _frames.VISITED:
		label.add_theme_color_override('font_color', Color.WHITE)
	else:
		label.add_theme_color_override('font_color', Color.BLACK)


#------------------------------ Set Sprite/Label -------------------------------

## Sets sprite to START, or START_END if it's already END
func set_start_sprite():
	if anim_sprite.frame == _frames.BLANK or anim_sprite.frame == _frames.VISITED:
		set_sprite(_frames.START)
	elif anim_sprite.frame == _frames.END:
		set_sprite(_frames.START_END)


## Sets sprite to BLANK if it's START, or END if it's START_END
func unset_start_sprite():
	if anim_sprite.frame == _frames.START:
		set_sprite(_frames.BLANK)
	elif anim_sprite.frame == _frames.START_END:
		set_sprite(_frames.END)


## Sets sprite to END, or START_END if it's already START
func set_end_sprite():
	if anim_sprite.frame == _frames.BLANK or anim_sprite.frame == _frames.VISITED:
		set_sprite(_frames.END)
	elif anim_sprite.frame == _frames.START:
		set_sprite(_frames.START_END)


## Sets sprite to BLANK if it's END, or START if it's START_END
func unset_end_sprite():
	if anim_sprite.frame == _frames.END:
		set_sprite(_frames.BLANK)
	elif anim_sprite.frame == _frames.START_END:
		set_sprite(_frames.START)


#--------------------------- Signal Responses ----------------------------------

func _on_button_down():
	marker_button_down.emit(_station)


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
