extends Node

# Sim mode containers
@export var single_run_mode_container : Node
@export var batch_mode_container : Node

# Batch mode containers
@export var fixed_mode_container : Node
@export var precision_mode_container : Node

func _on_param_controller_sim_and_batch_modes_initialized(sim_mode, batch_mode):
	_on_sim_mode_selected(sim_mode)
	_on_batch_mode_selected(batch_mode)

func _on_sim_mode_selected(index):
	if index == 0:
		single_run_mode_container.visible = true
		batch_mode_container.visible = false
	elif index == 1:
		single_run_mode_container.visible = false
		batch_mode_container.visible = true
		
func _on_batch_mode_selected(index):
	if index == 0:
		precision_mode_container.visible = true
		fixed_mode_container.visible = false
	elif index == 1:
		precision_mode_container.visible = false
		fixed_mode_container.visible = true
