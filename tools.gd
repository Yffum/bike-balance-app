extends Node

#----------------------- Global Tool Functions ---------------------------

func load_json_array(path: String) -> Array:
	# Get data
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("Failed to open JSON file: %s" % path)
		return []
	# Convert to array
	var content := file.get_as_text()
	var result: Variant = JSON.parse_string(content)
	if result == null:
		push_error("Failed to parse JSON.")
		return []
	# Ensure data is array
	if result is Array:
		return result as Array
	else:
		push_error("JSON data is not a Array.")
		return []

func load_json_dict(path: String) -> Dictionary:
	# Get data
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("Failed to open JSON file: %s" % path)
		return {}
	# Convert to dict
	var content := file.get_as_text()
	var result: Variant = JSON.parse_string(content)
	if result == null:
		push_error("Failed to parse JSON.")
		return {}
	# Ensure data is dict
	if result is Dictionary:
		return result as Dictionary
	else:
		push_error("JSON data is not a Dictionary.")
		return {}
