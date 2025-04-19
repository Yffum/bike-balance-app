extends Node

var log_count : int

func _ready():
	log_count = get_file_count(Tools.LOGS_PATH)

func get_file_count(directory_path: String) -> int:
	var dir = DirAccess.open(directory_path)
	var file_count = 0
	if dir != null:
		dir.list_dir_begin()  # Start iterating the directory
		while true:
			var file_name = dir.get_next()
			if file_name == "":
				break  # End of the directory listing
			# Check if it's a file (not a directory)
			if not dir.current_is_dir():
				file_count += 1
		dir.list_dir_end()  # Close the directory listing
	else:
		print("Failed to open directory.")
	return file_count
