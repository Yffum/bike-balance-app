extends Node

@export var param_ctrl : Node
var python_command_str = 'python3'

func _ready():
	pass


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func run_simulation():
	var python_script_path = ProjectSettings.globalize_path('res://simulate.py')
	var output := []
	var error := []
	var exit_code := OS.execute(python_command_str, [python_script_path], output)

	print('Exit code:', exit_code)
	print('Output:\n', '\n'.join(output))


func _on_run_button_pressed():
	param_ctrl.save_params()
	run_simulation() # Replace with function body.
