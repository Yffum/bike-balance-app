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
# Single Run
@export var seed_text : LineEdit
@export var seed_checkbox : CheckBox
@export var seed_setter: Node
# Batch
@export var confidence_level : SpinBox
@export var parallel_batch_size : SpinBox
@export var batch_mode : OptionButton
# Fixed sample size
@export var batch_size : SpinBox
@export var minimum_sample_size : SpinBox
@export var relative_margin_of_error : SpinBox
@export var absolute_margin_of_error : SpinBox
@export var max_runtime : SpinBox

func _on_tools_external_paths_set():
	_initialize()
	
func _initialize():
	var params = Tools.load_json_dict(Tools.USER_PARAMS_PATH)
	if params.is_empty():
		print("Warning: %s is corrupted. Loading default parameters." % Tools.USER_PARAMS_PATH)
		reset_params()
		save_params()
		print("Default parameters saved to %s" % Tools.USER_PARAMS_PATH)
	else:
		_load_params(params)
		print("Parameters loaded from %s" % Tools.USER_PARAMS_PATH)

## Loads parameters to interface
func _load_params(params : Dictionary):
	start_station.value = params['start_station']
	end_station.value = params['end_station']
	excursion_time.value = params['excursion_time']
	if params['agent_mode'] == "basic":
		agent_mode.selected = 0
	elif params['agent_mode'] == "smart":
		agent_mode.selected = 1
	# Set random seed if none given
	if params['seed'] == null:
		seed_text.text = str(randi())
	else:
		seed_text.text = str(int(params['seed']))
		# Check box if static seed given
		seed_checkbox.button_pressed = true
		seed_setter.visible = true
		
	batch_size.value = params['batch_size']
	empty_bias.value = params['empty_bias'] * 100
	full_bias.value = params['full_bias'] * 100
	warmup_time.value = params['warmup_time']

## Sets parameters to default values
func reset_params():
	var params = Tools.load_json_dict(Tools.DEFAULT_USER_PARAMS_PATH)
	_load_params(params)

## Saves parameters to file
func save_params():
	var params : Dictionary
	params['start_station'] = int(start_station.value)
	params['end_station'] = int(end_station.value)
	params['excursion_time'] = excursion_time.value
	# Set agent intelligence
	if agent_mode.selected == 0:
		params['agent_mode'] = "basic"
	elif agent_mode.selected == 1:
		params['agent_mode'] = "smart"
	# Check for static seed
	if seed_checkbox.button_pressed == true:  # Static seed set
		params['seed'] = int(seed_text.text)
	else:
		params['seed'] = null
	params['batch_size'] = int(batch_size.value)
	params['empty_bias'] = empty_bias.value / 100
	params['full_bias'] = full_bias.value / 100
	params['warmup_time'] = warmup_time.value
	# Save to file
	Tools.save_json(Tools.USER_PARAMS_PATH, params)
