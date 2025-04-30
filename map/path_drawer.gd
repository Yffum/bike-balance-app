extends Node2D

var _bike_paths : Array = []
var _walk_paths : Array = []

func _draw():
	for paths in _bike_paths:
		for path in paths:
			draw_polyline(path, Color.RED, 10.0)
