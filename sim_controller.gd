extends Node

@export var param_ctrl : Node
@export var log_label : RichTextLabel

var python_command_str = 'python3'

signal sim_complete

func _ready():
	pass
	

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

## Returns the filepath of the log
func run_simulation() -> String:
	var python_script_path = ProjectSettings.globalize_path('res://simulate.py')
	var output := []
	var error := []
	var exit_code := OS.execute(python_command_str, [python_script_path], output)

	#print('Exit code:', exit_code)
	#print('Output:\n', '\n'.join(output))
	return output[0]

func get_log(path : String) -> String:
	#print(DirAccess.get_files_at('res://logs'))
	path = ProjectSettings.globalize_path(path)
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		var fail_str = 'Failed to open log file: %s' % path
		return fail_str
	var content := file.get_as_text()
	return content

func _on_run_button_pressed():
	param_ctrl.save_params()
	var log_path = run_simulation() # Replace with function body.
	sim_complete.emit()
	log_path = 'res://' + log_path.replace('\\', '/')
	#await get_tree().create_timer(5.0).timeout
	log_label.text = get_log(log_path)
