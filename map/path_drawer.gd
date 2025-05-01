extends Node2D

# Initialized by map_controller
var _bike_paths : Array = []
var _walk_paths : Array = []

var _actions : Array = []

## True iff paths should be drawn
var _paths_are_active := false


func draw_excursion_paths():
	_paths_are_active = true
	queue_redraw()


func set_excursion_paths(actions : Array):
	_actions = actions


func hide_excursion_paths():
	_paths_are_active = false
	queue_redraw()


func _draw():
	if _paths_are_active:
		_draw_trip_paths()


func _draw_trip_paths():
	var LINE_WIDTH = 10.0
	var action : Dictionary
	var path : PackedVector2Array
	var color : Color
	for i in range(len(_actions)):
		action = _actions[i]
		# Determine path based on agent mode
		if action['agent_mode'] == 'wait':
			continue
		elif action['agent_mode'] == 'bike':
			path = _bike_paths[action['start_station']][action['end_station']]
			color = Color(1.0, 0.0, 0.0)
		elif action['agent_mode'] == 'walk':
			path = _walk_paths[action['start_station']][action['end_station']]
			color = Color(0.0, 0.0, 1.0)
		# Draw path for current action
		draw_polyline(path, color, LINE_WIDTH, true)



func _on_path_results_loaded(actions):
	_actions = actions
	queue_redraw()
