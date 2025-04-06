extends TabContainer

const PARAMS_FILEPATH = 'res://data/user_params.json'

# Use labels as parameters
var start_station
var end_station
var excursion_time
var agent_mode
var seed_text
var batch_size
var empty_bias
var full_bias
var warmup_time


# Called when the node enters the scene tree for the first time.
func _ready():
	_link_labels()
	_load_params()

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

## Gets the UI labels for parameters. Access values using ".value"
func _link_labels():
	var agent_container = $Agent/MarginContainer/VBoxContainer
	start_station = agent_container.get_node('StartStation/SpinBox')
	end_station = agent_container.get_node('EndStation/SpinBox')
	excursion_time = agent_container.get_node('ExcursionTime/SpinBox')
	# agent_mode value is ".selected" (int) 1 for smart, 0 for basic
	agent_mode = agent_container.get_node('AgentMode/OptionButton')
	var sim_container = $Simulation/MarginContainer/VBoxContainer
	seed_text = sim_container.get_node('SeedController/SetSeed/LineEdit')
	# Seed value is ".text"
	batch_size = sim_container.get_node('BatchSize/SpinBox')
	empty_bias = sim_container.get_node('EmptyBias/SpinBox')
	full_bias = sim_container.get_node('FullBias/SpinBox')
	warmup_time = sim_container.get_node('WarmupTime/SpinBox')

## Loads parameters from files
func _load_params():
	var params = Tools.load_json_dict(PARAMS_FILEPATH)
	print(params)
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
		seed_text.text = str(params['seed'])
		# Check box if static seed given
		var checkbox = $Simulation/MarginContainer/VBoxContainer/SeedController/ToggleSeed/CheckBox
		checkbox.button_pressed = true
		var seed_setter = $Simulation/MarginContainer/VBoxContainer/SeedController/SetSeed
		seed_setter.visible = true
		
	batch_size.value = params['batch_size']
	empty_bias.value = params['empty_bias']
	full_bias.value = params['full_bias']
	warmup_time.value = params['warmup_time']
