extends Node



func _init():
	var scale := 0.5
	ProjectSettings.set_setting("gui/theme/default_theme_scale", scale)
	ThemeDB.fallback_base_scale = scale
	print('scale set')
