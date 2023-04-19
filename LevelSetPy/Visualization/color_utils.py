__all__ = ["cm_colors", "cmaps"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# from colorspacious import cspace_converter
from collections import OrderedDict

cmaps = OrderedDict()


cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']

cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

cmaps['Sequential (2)'] = [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']

cmaps['Diverging'] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']

cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']

cmaps['Miscellaneous'] = [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar']

# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))

all_cmaps = []
for k, v in cmaps.items():
    all_cmaps += v

cm_colors = [
    "Accent", "Accent_r", "Blues",
    "Blues_r", "BrBG", "BrBG_r", "BuGn",
    "BuGn_r", "BuPu", "BuPu_r", "CMRmap",
    "CMRmap_r", "Dark2", "Dark2_r", "GnBu",
    "GnBu_r", "Greens", "Greens_r", "Greys",
    "Greys_r", "OrRd", "OrRd_r", "Oranges",
    "Oranges_r", "PRGn", "PRGn_r", "Paired",
    "Paired_r", "Pastel1", "Pastel1_r", "Pastel2",
    "Pastel2_r", "PiYG", "PiYG_r", "PuBu",
    "PuBuGn", "PuBuGn_r", "PuBu_r", "PuOr",
    "PuOr_r", "PuRd", "PuRd_r", "Purples",
    "Purples_r", "RdBu", "RdBu_r", "RdGy",
    "RdGy_r", "RdPu", "RdPu_r", "RdYlBu",
    "RdYlBu_r", "RdYlGn", "RdYlGn_r", "Reds",
    "Reds_r", "Set1", "Set1_r", "Set2",
    "Set2_r", "Set3", "Set3_r", "Spectral",
    "Spectral_r", "Wistia", "Wistia_r", "YlGn",
    "YlGnBu", "YlGnBu_r", "YlGn_r", "YlOrBr",
    "YlOrBr_r",  "YlOrRd",  "YlOrRd_r",  "afmhot",
    "afmhot_r", "autumn", "autumn_r", "binary",
    "binary_r", "bone", "bone_r", "brg",
    "brg_r", "bwr", "bwr_r", "cividis",
    "cividis_r", "cmap_d", "cool", "cool_r",
    "coolwarm", "coolwarm_r", "copper", "copper_r",
    "cubehelix", "cubehelix_r", "flag_r", "gist_earth",
    "gist_earth_r", "gist_gray", "gist_gray_r", "gist_heat",
    "gist_heat_r", "gist_ncar", "gist_ncar_r", "gist_rainbow",
    "gist_rainbow_r", "gist_stern", "gist_stern_r", "gist_yarg",
    "gist_yarg_r", "gnuplot", "gnuplot2", "gnuplot2_r", "gnuplot_r", "gray",
    "gray_r", "hot", "hot_r", "hsv", "hsv_r",
    "inferno", "inferno_r", "jet", "jet_r", "magma",
    "magma_r", "nipy_spectral","nipy_spectral_r","ocean","ocean_r","pink",
    "pink_r",  "plasma",  "plasma_r",  "prism",  "prism_r",  "rainbow",
    "rainbow_r", "seismic", "seismic_r", "spring", "spring_r", "summer",
    "summer_r", "tab10", "tab10_r", "tab20", "tab20_r",
    "tab20b", "tab20b_r", "tab20c", "tab20c_r", "terrain",
    "terrain_r", "turbo", "turbo_r", "twilight",
    "twilight_r", "twilight_shifted", "twilight_shifted_r", "viridis",
    "viridis_r", "winter", "winter_r"
    ]
