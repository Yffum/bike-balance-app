extends Node

var Marker = preload("res://map/marker/marker.tscn")
var markers : Node  # markers container

func _ready():
	markers = $Markers
	inst_markers()

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func inst_markers():
	var marker = Marker.instantiate()
	marker.position = Vector2(1775*2, 1594*2)
	markers.add_child(marker)
	
	
