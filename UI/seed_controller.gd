extends VBoxContainer

var checkbox : Node
var seed_setter: Node

# Called when the node enters the scene tree for the first time.
func _ready():
	checkbox = $ToggleSeed/CheckBox
	seed_setter = $SetSeed

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass


func _on_check_box_pressed():
	seed_setter.visible = checkbox.button_pressed
	

func _on_button_pressed():
	seed_setter.get_node('LineEdit').text = str(randi())
