extends Node

@export var param_ctrl : Node
@export var log_label : RichTextLabel
@export var run_button : Button
@export var spinner : Spinner
@export var log_tab : Node

var python_command_str = 'python3'

var thread : Thread

signal station_results_loaded(results : Dictionary)
signal station_batch_results_loaded()
signal path_results_loaded(actions)
signal batch_results_loaded()

func _notification(what):
	# On application close:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		# Save parameters
		param_ctrl.save_user_params()
		# Prune logs and results
		_delete_oldest_logs()
		_delete_oldest_results()
		# Quit
		get_tree().quit()  # default behavior


func _ready():
	# Set interface scale
	get_tree().root.content_scale_factor = 1.0
	thread = Thread.new()


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
			DirAccess.remove_absolute(full_path)
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
			DirAccess.remove_absolute(full_path)
		else:
			print("File not found:", full_path)


## Returns the filepath of the log
func run_simulation() -> String:
	var python_script_path = Tools.SIM_SCRIPT_PATH
	var output := []
	var exit_code := OS.execute(python_command_str, [python_script_path, '--frontend'], output, false, false)
	var stdout : String = output[0]
	return stdout


func _process_simulation():
	var results_path = run_simulation()
	call_deferred('_handle_sim_end', results_path)


func _handle_sim_end(results_path : String):
	print('results path:', results_path)
	results_path = results_path.replace('\\', '/')
	results_path = results_path.replace('external/', '')
	results_path = Tools.EXTERNAL_DIR.path_join(results_path)
	print('results path:', results_path)
	var results : Dictionary = Tools.load_json_dict(results_path)
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
	
	thread.wait_to_finish()
	# Enable run button, set spinner
	run_button.disabled = false
	run_button.text = 'Run Simulation'
	spinner.status = 3  # Success
	# Open log tab
	log_tab.visible = true


func _on_run_button_pressed():
	# Save parameters to file
	param_ctrl.save_user_params()
	# Disable run button, start spinner
	run_button.disabled = true
	run_button.text = 'Running...'
	spinner.visible = true
	spinner.status = 1  # Spinning
	# Run simulation in thread
	thread.start(_process_simulation)


func _exit_tree():
	thread.wait_to_finish()	


func _show_file(filepath : String):
	OS.shell_show_in_file_manager(filepath)
