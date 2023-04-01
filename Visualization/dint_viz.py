__all__ = ["DoubleIntegratorVisualizer"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import os
import time
import numpy as np
from os.path import join, expanduser
from skimage import measure
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class DoubleIntegratorVisualizer(object):
	def __init__(self, params=None):
		"""
			Class DoubleIntegratorVisualizer:

			This class expects to be constantly given values to plot in realtime.
			It assumes the values are an array and plots different indices at different
			colors according to the spectral colormap.

			Inputs:
				params: Bundle Type  with fields as follows:

				Bundle({"grid": obj.grid,
						'disp': True,
						'labelsize': 16,
						'labels': "Initial 0-LevelSet",
						'linewidth': 2,
						'data': data,
						'elevation': args.elevation,
						'azimuth': args.azimuth,
						'mesh': init_mesh,
						'init_conditions': False,
						'pause_time': args.pause_time,
						'level': 0, # which level set to visualize
						'winsize': (16,9),
						'fontdict': Bundle({'fontsize':12, 'fontweight':'bold'}),
						"savedict": Bundle({"save": False,
										"savename": "rcbrt",
										"savepath": "../jpeg_dumps/rcbrt"})
						})
		"""
		if params.winsize:
			self._fig = plt.figure(figsize=params.winsize)
			self.winsize=params.winsize
		else:
			self._fig = params.fig

		self.grid = params.grid

		# move data to cpu
		# self.grid.xs = [self.grid.xs[i].get() for i in range(self.grid.dim) \
		#   					if isinstance(self.grid.xs[i], cp.ndarray) else \
		# 						self.grid.xs[i]]

		self._gs  = gridspec.GridSpec(1, 2, self._fig)

		if self.grid.dim<=2:
			self._ax  = [plt.subplot(self._gs[i]) for i in [0, 1]]
		else:
			self._ax  = [plt.subplot(self._gs[i], projection='3d') for i in [0, 1]]


		self._init = False
		self.params = params

		if self.params.savedict.save and not os.path.exists(self.params.savedict.savepath):
			os.makedirs(self.params.savedict.savepath)

		if not 'fontdict' in self.params.__dict__.keys() and  self.params.fontdict is None:
			self._fontdict = {'fontsize':18, 'fontweight':'bold'}

		if np.any(params.mesh):
			self.init(params.mesh)#, params.pgd_mesh)
			self._init = True

		self._fig.canvas.draw()
		self._fig.canvas.flush_events()

	def init(self, mesh=None): #, pgd_mesh=None):
		"""
			Plot the initialize target set mesh.
			Inputs:
				data: marching cubes mesh
		"""
		cm = plt.get_cmap('rainbow')

		self._ax[0].grid('on')
		self._ax[0].axes.get_xaxis().set_ticks([])
		self._ax[0].axes.get_yaxis().set_ticks([])

		self._ax[1].axes.get_xaxis().set_ticks([])
		self._ax[1].axes.get_yaxis().set_ticks([])

		if self.grid.dim==3:

			self._ax[0].view_init(elev=self.params.elevation, azim=self.params.azimuth)
			self._ax[1].view_init(elev=self.params.elevation, azim=self.params.azimuth)

			self._ax[0].add_collection3d(mesh.mesh)


			xlim = (mesh.verts[:, 0].min(), mesh.verts[:,0].max())
			ylim = (mesh.verts[:, 1].min(), mesh.verts[:,1].max())
			zlim = (mesh.verts[:, 2].min(), mesh.verts[:,2].max())

			self._ax[0].set_xlim3d(*xlim)
			self._ax[0].set_ylim3d(*ylim)
			self._ax[0].set_zlim3d(*zlim)

			self._ax[0].set_xlabel("X", fontdict = self.params.fontdict.__dict__)
			self._ax[0].set_ylabel("Y", fontdict = self.params.fontdict.__dict__)
			self._ax[0].set_zlabel("Z", fontdict = self.params.fontdict.__dict__)
			self._ax[0].set_title(f"Initial {self.params.level}-Level Value Set", \
									fontweight=self.params.fontdict.fontweight)
			self._ax[1].set_title(f'BRT at {0} secs.', fontweight=self.params.fontdict.fontweight)
		elif self.grid.dim==2:
			self._ax[0].set_xlabel(rf'$x_1$', fontdict=self.params.fontdict.__dict__)
			self._ax[0].set_ylabel(rf'$x_2$', fontdict=self.params.fontdict.__dict__)
			self._ax[0].set_title(f'Closed-Form TTR @ -T secs.', fontdict =self.params.fontdict.__dict__)
			self._ax[0].contour(self.grid.xs[0], self.grid.xs[1], mesh, colors='crimson')

			X, Y = mesh[-1, ::len(mesh)//3], mesh[::len(mesh)//3, 1]
			U, V = Y, Y
			self._ax[0].quiver(X,Y, 2, 2, angles='xy')
			self._ax[0].set_xlim([-1.02, 1.02])
			self._ax[0].set_ylim([-1.01, 1.01])

			# # Decomposed level set
			# self._ax[1].contour(self.g_rom.xs[0], self.g_rom.xs[1], pgd_mesh, colors='magenta')

			# X, Y = pgd_mesh[-1, ::len(pgd_mesh)//2], pgd_mesh[::len(pgd_mesh)//2, 0]
			# U, V = Y, Y
			# self._ax[1].quiver(X,Y, 5, 5, angles='xy')

			# self._ax[1].set_xlabel(rf'$x_1$', fontdict=self.params.fontdict.__dict__)
			# self._ax[1].set_ylabel(rf'$x_2$', fontdict=self.params.fontdict.__dict__)
			# self._ax[1].set_title(f'Lax-Friedrichs Approx. @ -T secs.', fontdict =self.params.fontdict.__dict__)
			# self._ax[1].contour(self.grid.xs[0], self.grid.xs[1], mesh, colors='red')

			# self._ax[1].set_xlim([-1.02, 1.02])
			# self._ax[1].set_ylim([-1.01, 1.01])

	def update_tube(self, amesh, ls_mesh, time_step, delete_last_plot=False):
		"""
			Inputs:
				data - BRS/BRT data.
				amesh - level sets mesh of the analytic TTR mesh.
				ls_mesh - zero-level set mesh of the levelset tb TTR mesh.
				time_step - The time step at which we solved  this BRS/BRT.
				delete_last_plot - Whether to clear scene before updating th plot.
		"""
		self._ax[1].grid('on')

		self._ax[1].axes.get_xaxis().set_ticks([])
		self._ax[1].axes.get_yaxis().set_ticks([])


		if self.grid.dim==3:
			self._ax[1].add_collection3d(amesh.mesh)

			self._ax[1].add_collection3d(amesh.mesh)
			self._ax[1].view_init(elev=self.params.elevation, azim=self.params.azimuth)

			xlim = (amesh.verts[:, 0].min(), amesh.verts[:,0].max())
			ylim = (amesh.verts[:, 1].min(), amesh.verts[:,1].max())
			zlim = (amesh.verts[:, 2].min(), amesh.verts[:,2].max())

			self._ax[1].set_xlim3d(*xlim)
			self._ax[1].set_ylim3d(*ylim)
			self._ax[1].set_zlim3d(*zlim)

			self._ax[1].set_xlabel("X", fontdict = self.params.fontdict.__dict__)
			self._ax[1].set_title(f'BRT at {time_step}.', fontweight=self.params.fontdict.fontweight)

		elif self.grid.dim==2:
			self._ax[0].cla() if delete_last_plot else self._ax[0].cla()
			CS1 = self._ax[0].contour(self.grid.xs[0], self.grid.xs[1], amesh, linewidths=3,  colors='crimson')
			self._ax[0].grid('on')
			self._ax[0].set_xlabel(rf'$x_1$', fontdict=self.params.fontdict.__dict__)
			self._ax[0].set_ylabel(rf'$x_2$', fontdict=self.params.fontdict.__dict__)
			self._ax[0].set_title(f'Closed-Form TTR//{time_step} secs.', fontdict =self.params.fontdict.__dict__)

			self._ax[0].set_xlim([-1.02, 1.02])
			self._ax[0].set_ylim([-1.01, 1.01])
			self._ax[0].set_xticks([-0.75, 0.0, 1.01])
			self._ax[0].set_yticks([-0.75, 0.0, 1.01])
			self._ax[0].tick_params(axis='both', which='major', labelsize=28)
			self._ax[0].tick_params(axis='both', which='minor', labelsize=18)

			self._ax[0].clabel(CS1, CS1.levels, inline=True, fmt=self.fmt, fontsize=self.params.fontdict.fontsize)

			self._ax[1].cla() if delete_last_plot else self._ax[1].cla()
			CS1 = self._ax[1].contour(self.grid.xs[0], self.grid.xs[1], ls_mesh, linewidths=3,  colors='blue')
			self._ax[1].grid('on')
			self._ax[1].set_xlabel(rf'$x_1$', fontdict=self.params.fontdict.__dict__)
			self._ax[1].set_ylabel(rf'$x_2$', fontdict=self.params.fontdict.__dict__)
			self._ax[1].set_title(f'LF TTR@{time_step} secs.', fontdict=self.params.fontdict.__dict__)


			self._ax[1].set_xlim([-1.02, 1.02])
			self._ax[1].set_ylim([-1.01, 1.01])
			self._ax[1].set_xticks([-0.75, 0.0, 1.01])
			self._ax[1].set_yticks([-0.75, 0.0, 1.01])
			self._ax[1].tick_params(axis='both', which='major', labelsize=28)
			self._ax[1].tick_params(axis='both', which='minor', labelsize=18)

			# self._ax[1].clabel(CS2, CS2.levels, inline=True, fmt=self.fmt, fontsize=self.params.fontdict.fontsize)

		plt.tight_layout()
		f = plt.gcf()
		f.savefig(join(expanduser("~"),"Documents/Papers/Safety/PGDReach", f"figures/dint_ttr_{time_step}.jpg"),
			bbox_inches='tight',facecolor='None')
		self.draw()
		time.sleep(self.params.pause_time)

	def fmt(self, x):
		s = f"{x:.2f}"
		if s.endswith("0"):
			s = f"{x:.0f}"
		return rf"{s} \s" if plt.rcParams["text.usetex"] else f"{s}"

	def add_legend(self, linestyle, marker, color, label):
		self._ax_legend.plot([], [], linestyle=linestyle, marker=marker,
				color=color, label=label)
		self._ax_legend.legend(ncol=2, mode='expand', fontsize=self.params.fontdict.fontsize)

	def draw(self, ax=None):
		self._fig.canvas.draw()
		self._fig.canvas.flush_events()
