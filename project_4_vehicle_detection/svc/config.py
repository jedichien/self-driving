feature_extraction_params = {
    'resize_h': 64,
    'resize_w': 64,
    'color_space': 'YCrCb',
    'orient': 9,
    'pix_per_cell': 8,
    'cell_per_block': 2,
    'hog_channel': 'ALL',
    'spatial_size': (32, 32), # spatial binning dimensions
    'hist_bins': 16,
    'spatial_feature': True,
    'hist_feature': True,
    'hog_feature': True,
}