extends Node

# Sim mode containers
@export var single_run_mode_container : Node
@export var batch_mode_container : Node

# Batch mode containers
@export var fixed_mode_container : Node
@export var precision_mode_container : Node

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass


func _on_batch_mode_selected(index):
	if index == 0:
		precision_mode_container.visible = true
	elif index == 1:
		fixed_mode_container.visible = true


func _on_sim_mode_selected(index):
	if index == 0:
		single_run_mode_container.visible = true
		batch_mode_container.visible = false
	elif index == 1:
		single_run_mode_container.visible = false
		batch_mode_container.visible = true
		
