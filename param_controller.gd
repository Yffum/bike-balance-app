extends Node

#--------- Agent Parameters --------
@export var start_station : SpinBox
@export var end_station : SpinBox
@export var excursion_time : SpinBox
@export var agent_mode : OptionButton

#------ Simulation Parameters ------
@export var full_bias : SpinBox
@export var empty_bias : SpinBox
@export var warmup_time : SpinBox
@export var sim_mode : OptionButton
#Single Run
@export var seed_text : LineEdit
@export var seed_checkbox : CheckBox
@export var seed_setter: Node
# Batch
@export var confidence_level : SpinBox
@export var parallel_batch_size : SpinBox
@export var batch_mode : OptionButton
# Fixed sample size
@export var batch_size : SpinBox
# Precision based
@export var min_sample_size : SpinBox
@export var relative_margin_of_error : SpinBox
@export var absolute_margin_of_error : SpinBox
@export var max_runtime : SpinBox

#------------------ Station -------------------
@export var station_spinbox : SpinBox
@export var no_station_selected : Label
@export var station_content : Node
@export var is_start_station : CheckButton
@export var is_end_station : CheckButton
# Parameters
@export var station_name : Label
@export var station_id : Label
@export var rent_rate : Label
@export var return_rate : Label
@export var initial_bike_count : Label
@export var max_bike_count : Label
@export var initial_incentive : Label
@export var initial_incentive_type : Label
# Results
@export var final_bike_count : Label 
@export var final_incentive : Label
@export var agent_rent_count : Label
@export var agent_return_count : Label
@export var results_incentive_label : Label
@export var start_end_station_label : Label

#----------- Panels ----------
@export var results_tab : Node
@export var station_results_panel : Node
@export var start_end_station_switches : Node

var sim_params : Dictionary
var incentives : Array  # incentives[<station>][<bike_count>]
var station_names : Array
var station_ids : Array
var results_are_loaded := false
var station_results : Dictionary

signal sim_and_batch_modes_initialized(sim_mode : int, batch_mode : int)
signal station_selected(station : int)
signal start_station_set(station : int)
signal end_station_set(station : int)


func _on_tools_external_paths_set():
	sim_params = Tools.load_json_dict(Tools.SIM_PARAMS_PATH)
	incentives = Tools.load_json_array(Tools.INCENTIVES_PATH)
	station_names = Tools.load_json_array(Tools.STATION_NAMES_PATH)
	station_ids = Tools.load_json_array(Tools.STATION_IDS_PATH)
	_initialize_user_params()
	station_spinbox.get_child(0, true).text = 'Select station'


func _initialize_user_params():
	var params = Tools.load_json_dict(Tools.USER_PARAMS_PATH)
	if params.is_empty():
		print("Warning: Unable to load parameters from %s " % Tools.USER_PARAMS_PATH)
		print("Loading default parameters.")
		reset_user_params()
		save_user_params()
		print("Default parameters saved to %s" % Tools.USER_PARAMS_PATH)
	else:
		_load_user_params(params)
		print("Parameters loaded from %s" % Tools.USER_PARAMS_PATH)


## Loads parameters to interface
func _load_user_params(params : Dictionary):
	#--------- Agent Parameters --------
	_set_start_station(params['start_station'])
	_set_end_station(params['end_station'])
	excursion_time.value = params['excursion_time']
	if params['agent_mode'] == "basic":
		agent_mode.selected = 0
	elif params['agent_mode'] == "smart":
		agent_mode.selected = 1
	#------ Simulation Parameters ------
	empty_bias.value = params['empty_bias'] * 100  # factor to percent
	full_bias.value = params['full_bias'] * 100  # factor to percent
	warmup_time.value = params['warmup_time']
	if params['sim_mode'] == 'single_run':
		sim_mode.selected = 0
	elif params['sim_mode'] == 'batch':
		sim_mode.selected = 1
	# Single Run
	# Set random seed if none given
	if params['seed'] == null or params['seed'] == 0:
		seed_text.text = str(randi())
	else:
		seed_text.text = str(int(params['seed']))
	seed_checkbox.button_pressed = params['use_static_seed']
	seed_setter.visible = params['use_static_seed']
	# Batch
	confidence_level.value = params['confidence_level'] * 100  # factor to percent
	parallel_batch_size.value = params['parallel_batch_size']
	if params['batch_mode'] == 'precision_based':
		batch_mode.selected = 0
	elif params['batch_mode'] == 'fixed_sample_size':
		batch_mode.selected = 1
	# Fixed sample size
	batch_size.value = params['batch_size']
	# Precision based
	min_sample_size.value = params['min_sample_size']
	relative_margin_of_error.value = params['relative_margin_of_error'] * 100  # factor to percent
	absolute_margin_of_error.value = params['absolute_margin_of_error']
	max_runtime.value = params['max_runtime'] / 60.0  # seconds to minutes
	
	# Set up interface
	sim_and_batch_modes_initialized.emit(sim_mode.selected, batch_mode.selected)


## Sets parameters to default values
func reset_user_params():
	var params = Tools.load_json_dict(Tools.DEFAULT_USER_PARAMS_PATH)
	_load_user_params(params)


## Saves parameters to file
func save_user_params():
	var params : Dictionary
	#--------- Agent Parameters --------
	params['start_station'] = int(start_station.value)
	params['end_station'] = int(end_station.value)
	params['excursion_time'] = excursion_time.value
	# Set agent intelligence
	if agent_mode.selected == 0:
		params['agent_mode'] = "basic"
	elif agent_mode.selected == 1:
		params['agent_mode'] = "smart"
	#------ Simulation Parameters ------
	params['full_bias'] = full_bias.value / 100  # percent to factor
	params['empty_bias'] = empty_bias.value / 100  # percent to factor
	params['warmup_time'] = warmup_time.value
	if sim_mode.selected == 0:
		params['sim_mode'] = 'single_run'
	elif sim_mode.selected == 1:
		params['sim_mode'] = 'batch'
	# Single Run
	params['use_static_seed'] = seed_checkbox.button_pressed
	params['seed'] = int(seed_text.text)
	# Batch
	params['confidence_level'] = confidence_level.value / 100 # percent to factor
	params['parallel_batch_size'] = int(parallel_batch_size.value)
	if batch_mode.selected == 0:
		params['batch_mode'] = 'precision_based'
	elif batch_mode.selected == 1:
		params['batch_mode'] = 'fixed_sample_size'
	# Fixed sample size
	params['batch_size'] = int(batch_size.value)
	# Precision based
	params['min_sample_size'] = int(min_sample_size.value)
	params['relative_margin_of_error'] = relative_margin_of_error.value / 100 # percent to factor
	params['absolute_margin_of_error'] = absolute_margin_of_error.value
	params['max_runtime'] = max_runtime.value * 60  # minutes to seconds

	# Save to file
	Tools.save_json(Tools.USER_PARAMS_PATH, params)


## Sets the station parameters in the UI for the given station
func set_station_params(station : int):
	if results_are_loaded:
		_set_station_results(station)
	# Set checkbox if station is start
	if station == start_station.value:
		is_start_station.button_pressed = true
		is_start_station.disabled = true
	else:
		is_start_station.button_pressed = false
		is_start_station.disabled = false
	# Set checkbox if station is end
	if station == end_station.value:
		is_end_station.button_pressed = true
		is_end_station.disabled = true
	else:
		is_end_station.button_pressed = false
		is_end_station.disabled = false
	var PRECISION = 0.01
	station_name.text = station_names[station]
	station_id.text = str(int(station_ids[station]))
	rent_rate.text = str(snappedf(sim_params['rent_rates'][station], PRECISION))
	return_rate.text = str(snappedf(sim_params['return_rates'][station], PRECISION))
	var bike_count : int = sim_params['initial_bike_counts'][station]
	initial_bike_count.text = str(bike_count)
	max_bike_count.text = str(int(sim_params['capacities'][station]))
	var incentive : float = snappedf(incentives[station][bike_count], PRECISION)
	initial_incentive.text = str(incentive)
	var incentive_str = ''
	if incentive < 0:
		incentive_str = '(Rent)'
	elif incentive > 0: 
		incentive_str = '(Return)'
	initial_incentive_type.text = incentive_str


## Reveals stations results in the station tab
func show_station_results():
	station_results_panel.visible = results_are_loaded
	start_end_station_switches.visible = false


## Hides station results in the station tab
func hide_station_results():
	station_results_panel.visible = false
	start_end_station_switches.visible = true


## Set station results in the UI for the given station
func _set_station_results(station : int):
	var capacity := str(int(sim_params['capacities'][station]))
	var bike_count := str(int(station_results['final_bike_counts'][station]))
	final_bike_count.text = bike_count + ' / ' + capacity
	final_incentive.text = str(station_results['final_incentives'][station])
	agent_rent_count.text = str(int(station_results['rent_counts'][station]))
	agent_return_count.text = str(int(station_results['return_counts'][station]))
	# Set incentive label
	if station_results['final_incentives'][station] < 0:
		results_incentive_label.text = '(Rent)'
	elif station_results['final_incentives'][station] > 0: 
		results_incentive_label.text = '(Return)'
	else:
		results_incentive_label.text = ''
	# Set start station label
	start_end_station_label.visible = true
	if station == station_results['start_station'] and station == station_results['end_station']:
		start_end_station_label.text = "(Start/End Station)"
	elif station == station_results['start_station']:
		start_end_station_label.text = "(Start Station)"
	elif station == station_results['end_station']:
		start_end_station_label.text = "(End Station)"
	else:
		start_end_station_label.visible = false


func _set_start_station(station : int):
	start_station.value = station
	start_station_set.emit(station)
	set_station_params(station_spinbox.value)


func _set_end_station(station: int):
	end_station.value = station
	end_station_set.emit(station)
	set_station_params(station_spinbox.value)


## Update station params when map station selected
func _on_map_controller_station_selected(station):
	no_station_selected.visible = false
	station_content.visible = true
	station_spinbox.value = station
	set_station_params(station)


## Update station params when spinbox value changed
func _on_station_spinbox_value_changed(value):
	no_station_selected.visible = false
	station_content.visible = true
	set_station_params(value)


## Activate station panel when spinbox is clicked
func _on_station_spinbox_gui_input(event):
	if event is InputEventMouseButton:
		no_station_selected.visible = false
		station_content.visible = true
		set_station_params(station_spinbox.value)
		station_selected.emit(station_spinbox.value)


func _on_start_station_spinbox_value_changed(value):
	_set_start_station(value)


func _on_end_station_spinbox_value_changed(value):
	_set_end_station(value)


## Set start station to selected station
func _on_is_start_station_toggled(toggled_on):
	if toggled_on:
		is_start_station.disabled = true
		_set_start_station(station_spinbox.value)


## Set End station to selected station
func _on_is_end_station_toggled(toggled_on):
	if toggled_on:
		is_end_station.disabled = true
		_set_end_station(station_spinbox.value)


func _on_station_results_loaded(results):
	results_are_loaded = true
	station_results = results
	_set_station_results(station_spinbox.value)
	show_station_results()


func _on_batch_results_loaded():
	results_are_loaded = false
	show_station_results()


func _on_settings_tab_changed(tab):
	#if results_tab.visible == true:
	if tab == 2:
		show_station_results()
	else:
		hide_station_results()
