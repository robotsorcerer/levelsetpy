__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

labelsize=18
linewidth=6
fontdict={'fontsize':12, 'fontweight':'bold'}
winsize =(16, 9)
colors = ['blue', 'red', 'yellow', 'orange', \
            'green', 'black', 'cyan', 'magenta']

import numpy as np

def buffered_axis_limits(amin, amax, buffer_factor=1.05):
	"""
	Increases the range (amin, amax) by buffer_factor on each side
	and then rounds to precision of 1/10th min or max.
	Used for generating good plotting limits.
	For example (0, 100) with buffer factor 1.1 is buffered to (-10, 110)
	and then rounded to the nearest 10.
	"""
	diff = amax - amin
	amin -= (buffer_factor-1)*diff
	amax += (buffer_factor-1)*diff
	magnitude = np.floor(np.log10(np.amax(np.abs((amin, amax)) + 1e-100)))
	precision = np.power(10, magnitude-1)
	amin = np.floor(amin/precision) * precision
	amax = np.ceil (amax/precision) * precision
	return (amin, amax)
