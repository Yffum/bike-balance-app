extends VBoxContainer

var checkbox : Node
var seed_setter: Node

func _ready():
	checkbox = $ToggleSeed/CheckBox
	seed_setter = $SetSeed

func _on_check_box_pressed():
	seed_setter.visible = checkbox.button_pressed
	

func _on_button_pressed():
	seed_setter.get_node('LineEdit').text = str(randi())
