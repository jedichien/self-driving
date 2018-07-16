w, h, c = 200, 66, 3

config = {
    'input_width': w,
    'input_height': h,
    'input_channels': c,
    'batch_size': 256,
    'delta_correction': 0.18,
    'crop_height': range(20, 140),
    'augmentation_steer_sigma': 0.2,
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,
    'bias': 0.8,
    'max_speed': 30
}
