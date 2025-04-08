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
		
func load_json(path: String) -> Variant:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error('Failed to open JSON file: %s' % path)
		return null
	var content := file.get_as_text()
	return JSON.parse_string(content)

func save_json(path: String, data: Variant) -> void:
	# Open the file for writing
	var file = FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		push_error("Failed to open file for writing: %s" % path)
		return
	# Convert the data to a JSON string 
	var json_string: String
	if data is Dictionary or data is Array:
		json_string = JSON.stringify(data)
	else:
		push_error("Unsupported data type: %s" % data)
		return
	# Write the JSON string to the file
	file.store_string(json_string)
	file.close()
	print("File saved at: %s" % path)
