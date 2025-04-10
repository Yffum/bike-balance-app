extends Node

@export var param_ctrl : Node
@export var log_label : RichTextLabel
@export var run_button : Button
@export var spinner : Spinner
@export var log_tab : Node

var python_command_str = 'python3'

var thread : Thread

signal sim_complete

func _notification(what):
	# On application close:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		# Save parameters
		param_ctrl.save_params()
		# Prune logs
		_delete_oldest_logs()
		# Quit
		get_tree().quit()  # default behavior

func _ready():
	thread = Thread.new()

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

## Returns the content of the given log as a String
func get_log(path : String) -> String:
	#print(DirAccess.get_files_at('res://logs'))
	path = ProjectSettings.globalize_path(path)
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		var fail_str = 'Failed to open log file: %s' % path
		return fail_str
	var content := file.get_as_text()
	return content

## Keeps only `Tools.MAX_LOG_COUNT` most recent logs in the logs folder
func _delete_oldest_logs():
	# Open log directory
	var dir = DirAccess.open(Tools.LOGS_PATH)
	if dir == null:
		push_error("Cannot open directory: %s" % Tools.LOGS_PATH)
		return
	# Create log regex for new format: YYMMDD_HHMM_s<seed> or YYMMDD_HHMM_batch with optional -n suffix
	var regex = RegEx.new()
	regex.compile("^([0-9]{6}_[0-9]{4})_(s\\d+|batch)(?:-(\\d+))?\\.log$")
	# Search logs
	var logs = []
	dir.list_dir_begin()
	var file_name = dir.get_next()
	while file_name != "":
		if not dir.current_is_dir() and file_name.ends_with(".log"):
			var result = regex.search(file_name)
			if result:
				var base_time = result.get_string(1)
				var suffix_number = int(result.get_string(3)) if result.get_group_count() > 2 else 0
				logs.append({
					"file": file_name,
					"timestamp": base_time,
					"suffix_number": suffix_number
				})
		file_name = dir.get_next()
	dir.list_dir_end()
	# Report no logs found
	if logs.size() == 0:
		print("No log files found.")
		return
	# Sort by timestamp, then by numeric suffix
	logs.sort_custom(func(a, b):
		if a.timestamp == b.timestamp:
			return a.suffix_number < b.suffix_number
		return a.timestamp < b.timestamp
	)
	# Determine how many to delete
	var delete_count = logs.size() - Tools.MAX_LOG_COUNT
	if delete_count <= 0:
		print("No logs need to be deleted. Current count: %d, Max allowed: %d" % [logs.size(), Tools.MAX_LOG_COUNT])
		return
	# Remove oldest files
	for i in range(delete_count):
		var log_file = logs[i].file
		var full_path = Tools.LOGS_PATH.path_join(log_file)
		if FileAccess.file_exists(full_path):
			print("Deleting log:", log_file)
			DirAccess.remove_absolute(full_path)
		else:
			print("File not found:", full_path)

func _process_simulation():
	var log_path = run_simulation() # Replace with function body.
	call_deferred('_handle_sim_end', log_path)

func _handle_sim_end(log_path : String):
	log_path = 'res://' + log_path.replace('\\', '/')
	# Write log text to panel
	log_label.text = get_log(log_path)
	thread.wait_to_finish()
	# Enable run button, set spinner
	run_button.disabled = false
	run_button.text = 'Run Simulation'
	spinner.status = 3  # Success
	# Open log tab
	log_tab.visible = true

func _on_run_button_pressed():
	# Save parameters to file
	param_ctrl.save_params()
	# Disable run button, start spinner
	run_button.disabled = true
	run_button.text = 'Running...'
	spinner.visible = true
	spinner.status = 1  # Spinning
	# Run simulation in thread
	thread.start(_process_simulation)

func _exit_tree():
	thread.wait_to_finish()	
