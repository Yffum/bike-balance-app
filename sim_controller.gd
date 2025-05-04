extends Node

@export var param_ctrl : Node
@export var log_label : RichTextLabel
@export var run_button : Button
@export var spinner : Spinner
@export var _results_tab : Container
@export var _cancel_sim_button : Container
@export var _sim_feedback_label : Label

@export var _python_error_dialog : AcceptDialog
@export var _python_error_label : Control

@export var _python_interpreter_line_edit : LineEdit

signal station_results_loaded(results : Dictionary)
signal station_batch_results_loaded()
signal path_results_loaded(actions)
signal batch_results_loaded()

## True iff python script simulation is running
var _sim_is_running = false
## Access to the process stdin and stdout pipes for python
var _python_stdio : FileAccess
## Access to the python process error output
var _python_stderr : FileAccess
## Python process ID
var _python_pid : int


func _notification(what):
	# On application close:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		# Stop sim if running
		kill_simulation()
		# Save parameters
		param_ctrl.save_user_params()
		# Prune logs and results
		_delete_oldest_logs()
		_delete_oldest_results()
		# Quit
		get_tree().quit()  # default behavior


func _ready():
	# Set focus to run button
	run_button.grab_focus()


func _process(_delta):
	# Check if simulation is finished
	if _sim_is_running and not OS.is_process_running(_python_pid):
		# Hide cancel button
		_cancel_sim_button.visible = false
		# Get results filepath and process results
		_sim_is_running = false
		# Get results filepath from python
		var results_filepath := _python_stdio.get_as_text()
		results_filepath = results_filepath.replace('\\', '/')
		# Reformat based on external dir (for running in editor vs build)
		results_filepath = results_filepath.replace('external/', '')
		results_filepath = Tools.EXTERNAL_DIR.path_join(results_filepath)
		# Load results
		var results : Dictionary = Tools.load_json_dict(results_filepath)
		# If not results, report python error
		if results.is_empty():
			_process_sim_failure('Failed to retrieve simulation results')
			_python_error_dialog.visible = true
			var command_str := _python_interpreter_line_edit.text
			_python_error_label.text = (
				'> ' + command_str + ' ' + Tools.SIM_SCRIPT_PATH + '--frontend'
				+ '\n\n' + _python_stderr.get_as_text()
			)
		# Otherwise, process results
		else:
			_process_sim_results(results)


## Returns the content of the given log as a String
func get_log(path : String) -> String:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		var fail_str = 'Failed to open log file: %s' % path
		return fail_str
	var content := file.get_as_text()
	return content


## Keeps only `Tools.MAX_LOG_COUNT` most recent logs in the logs folder
func _delete_oldest_logs():
	print("Pruning logs...")
	# Open log directory
	var dir = DirAccess.open(Tools.LOGS_PATH)
	if dir == null:
		push_error("Cannot open directory: %s" % Tools.LOGS_PATH)
		return
	# Create log regex for YYMMDD_HHMM_s<seed> or YYMMDD_HHMM_batch with optional -n suffix
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
			OS.move_to_trash(full_path)
		else:
			print("File not found:", full_path)


## Keeps only `Tools.MAX_RESULTS_COUNT` most recent logs in the logs folder
func _delete_oldest_results():
	print("Pruning results...")
	# Open log directory
	var dir = DirAccess.open(Tools.RESULTS_PATH)
	if dir == null:
		push_error("Cannot open directory: %s" % Tools.RESULTS_PATH)
		return
	# Create log regex for YYMMDD_HHMM_s<seed> or YYMMDD_HHMM_batch with optional -n suffix
	var regex = RegEx.new()
	regex.compile("^([0-9]{6}_[0-9]{4})_(s\\d+|batch)(?:-(\\d+))?\\.json$")
	# Search logs
	var results = []
	dir.list_dir_begin()
	var file_name = dir.get_next()
	while file_name != "":
		if not dir.current_is_dir() and file_name.ends_with(".json"):
			var regex_result = regex.search(file_name)
			if regex_result:
				var base_time = regex_result.get_string(1)
				var suffix_number = int(regex_result.get_string(3)) if regex_result.get_group_count() > 2 else 0
				results.append({
					"file": file_name,
					"timestamp": base_time,
					"suffix_number": suffix_number
				})
		file_name = dir.get_next()
	dir.list_dir_end()
	# Report no results found
	if results.size() == 0:
		print("No log files found.")
		return
	# Sort by timestamp, then by numeric suffix
	results.sort_custom(func(a, b):
		if a.timestamp == b.timestamp:
			return a.suffix_number < b.suffix_number
		return a.timestamp < b.timestamp
	)
	# Determine how many to delete
	var delete_count = results.size() - Tools.MAX_RESULTS_COUNT
	if delete_count <= 0:
		print("No results need to be deleted. Current count: %d, Max allowed: %d" % [results.size(), Tools.MAX_RESULTS_COUNT])
		return
	# Remove oldest files
	for i in range(delete_count):
		var results_file = results[i].file
		var full_path = Tools.RESULTS_PATH.path_join(results_file)
		if FileAccess.file_exists(full_path):
			print("Deleting result:", results_file)
			OS.move_to_trash(full_path)
		else:
			print("File not found:", full_path)


## Runs simulation through separate python process and saves process info
func run_simulation() -> void:
	var python_script_path := Tools.SIM_SCRIPT_PATH
	var command_str := _python_interpreter_line_edit.text
	var process_info := OS.execute_with_pipe(command_str, [python_script_path, '--frontend'])
	# Python script successfully launched
	if not process_info.is_empty():
		print('Python script launched')
		_python_pid = process_info['pid']
		_python_stdio = process_info['stdio']
		_python_stderr = process_info['stderr']
		_sim_is_running = true
	else:
		print('Failed to launch ' + command_str + ' ' + python_script_path)
		_process_sim_failure('Simulation failed to launch')


## Stops simulation and adjusts interface
func cancel_simulation() -> void:
	# Hide cancel button and kill simulation
	_cancel_sim_button.visible = false
	run_button.grab_focus()
	kill_simulation()
	# Enable run button
	run_button.disabled = false
	run_button.text = 'Run Simulation'
	# Set spinner to warning
	spinner.status = 4  # Warning
	# Show feedback
	_sim_feedback_label.text = 'Simulation canceled'


## Stops simulation, or does nothing if simulation isn't running.
func kill_simulation() -> void:
	if _sim_is_running:
		_sim_is_running = false
		var error := OS.kill(_python_pid)
		if error == OK:
			print('Python process killed successfully.')
		else:
			print('Failed to kill Python process.')


## Updates interface after sim fails to complete
func _process_sim_failure(feedback_text : String) -> void:
	# Enable run button
	run_button.disabled = false
	run_button.text = 'Run Simulation'
	# Hide cancel button
	_cancel_sim_button.visible = false
	# Set spinner
	spinner.status = 5  # Error
	# Show feedback
	_sim_feedback_label.text = feedback_text


func _process_sim_results(results : Dictionary) -> void:
	# Write log text to panel
	log_label.text = results['report']
	# Save station results if single run
	if results['user_params']['sim_mode'] == 'single_run':
		var station_results := {
			'final_bike_counts' : results['data']['final_bike_counts'],
			'final_incentives' : results['data']['final_incentives'],
			'rent_counts' : results['data']['rent_counts'],
			'return_counts' : results['data']['return_counts'],
			'start_station' : results['user_params']['start_station'],
			'end_station' : results['user_params']['end_station'],
		}
		station_results_loaded.emit(station_results)
		var actions : Array = results['data']['actions']
		path_results_loaded.emit(actions)
	else:
		station_batch_results_loaded.emit()
		batch_results_loaded.emit()
	
	# Enable run button
	run_button.disabled = false
	run_button.text = 'Run Simulation'
	# Hide cancel button
	_cancel_sim_button.visible = false
	# Set spinner
	spinner.status = 3  # Success
	# Show feedback
	_sim_feedback_label.text = 'Simulation complete'
	# Open log tab
	_results_tab.visible = true


func _on_run_button_pressed():
	# Save parameters to file
	param_ctrl.save_user_params()
	# Disable run button
	run_button.disabled = true
	run_button.text = 'Running...'
	# Show cancel button
	_cancel_sim_button.visible = true
	# Start spinner
	spinner.visible = true
	spinner.status = 1  # Spinning
	# Wipe feedback label
	_sim_feedback_label.text = ''
	# Run simulation as separate process
	run_simulation()


func _show_file(filepath : String):
	OS.shell_show_in_file_manager(filepath)


func _on_cancel_sim_button_pressed():
	cancel_simulation()
