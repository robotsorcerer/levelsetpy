__all__ = ["ROCKETSVisualizer"]

import os
import time
import numpy as np
from skimage import measure
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from os.path import join, expanduser
from LevelSetPy.Utilities import isfield

class ROCKETSVisualizer(object):
	def __init__(self, params=None):
		"""
			Class ROCKETSVisualizer:

			This class expects to be constantly given np.s to plot in realtime.
			It assumes the np.s are an array and plots different indices at different
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
						'pause_time': args.pause_time,
						'level': 0, # which level set to visualize
						'winsize': (16,9),
						'fontdict': Bundle({'fontsize':12, 'fontweight':'bold'}),
						"savedict": Bundle({"save": False,
										"savename": "rcbrt",
										"savepath": "../jpeg_dumps/rcbrt"})
						})
		"""
		plt.ion()

		# Housekeeping
		if params.winsize:
			self.winsize=params.winsize
			self._fig = plt.figure(figsize=params.winsize)
		else:
			self._fig = plt.figure(figsize=(16,9))

		if params.savedict.save:
			self.fname = params.savedict.savepath

		self.grid = params.grid
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
			self._fontdict = {'fontsize':12, 'fontweight':'bold'}

		if np.any(params.mesh):
			self.init(params.mesh)
			self._init = True

		self._fig.canvas.draw()
		self._fig.canvas.flush_events()

	def init(self, mesh=None):
		"""
			Plot the initialize target set mesh.
			Inputs:
				data: marching cubes mesh
		"""
		self._ax[0].grid('on')

		self._ax[0].view_init(elev=self.params.elevation, azim=self.params.azimuth)
		self._ax[1].view_init(elev=self.params.elevation, azim=self.params.azimuth)

		self._ax[0].axes.get_xaxis().set_ticks([])
		self._ax[0].axes.get_yaxis().set_ticks([])

		self._ax[1].axes.get_xaxis().set_ticks([])
		self._ax[1].axes.get_yaxis().set_ticks([])

		if self.grid.dim==3:
			self._ax[0].add_collection3d(mesh.mesh)
			xlim, ylim, zlim = self.get_lims(mesh.verts)

			self._ax[0].set_xlim3d(*xlim)
			self._ax[0].set_ylim3d(*ylim)
			self._ax[0].set_zlim3d(*zlim)

			if isfield(self.params,  "title"):
				self._ax[0].set_title(self.params.title, fontdict=self.params.fontdict)
			else:
				self._ax[0].set_title(f'Initial {self.params.level} Tube.', fontdict=self.params.fontdict)

		elif self.grid.dim==2:
			self._ax[0].contourf(self.grid.xs[0], self.grid.xs[1], mesh.mesh, colors='cyan')
			self._ax[0].set_title(f'BRT\'s {self.params.level}-LevelSet.', \
									fontdict=self.params.fontdict)
		self._ax[0].set_xlabel(rf'$x$ (m)', fontdict=self.params.fontdict)
		self._ax[0].set_ylabel(rf'$z$ (m)', fontdict=self.params.fontdict)
		self._ax[0].set_zlabel(rf'$\theta$ (deg)',fontdict=self.params.fontdict)

	def update_tube(self, mesh_bundle, time_step, delete_last_plot=False):
		"""
			Inputs:
				mesh_bundle - zero-level set mesh of the BRT(S).
				time_step - The timne step at which we solved  this BRS/BRT.
				delete_last_plot - Whether to clear scene before updating th plot.

		"""
		self._ax[1].grid('on')
		self._ax[1].view_init(elev=self.params.elevation, azim=self.params.azimuth)

		self._ax[1].axes.get_xaxis().set_ticks([])
		self._ax[1].axes.get_yaxis().set_ticks([])

		if delete_last_plot:
			plt.cla()

		if self.grid.dim==3:
			self._ax[1].add_collection3d(mesh_bundle.mesh)
			xlim, ylim, zlim = self.get_lims(mesh_bundle.verts)

			self._ax[1].set_xlim3d(*xlim)
			self._ax[1].set_ylim3d(*ylim)
			self._ax[1].set_zlim3d(*zlim)

		elif len(self.grid.dim)==2:
			self._ax[1].contourf(self.grid.xs[0], self.grid.xs[1], mesh_bundle, colors='cyan')

		self._ax[1].set_xlabel(rf'$x$ (m)', fontdict=self.params.fontdict)
		self._ax[1].set_ylabel(rf'$z$ (m)', fontdict=self.params.fontdict)
		self._ax[1].set_zlabel(rf'$\theta$ (rad)',fontdict=self.params.fontdict)
		self._ax[1].set_title(f'BRT at {time_step} secs.', fontdict=self.params.fontdict)

		self.draw()
		time.sleep(self.params.pause_time)

	def get_lims(self, verts):
		xlim = (verts[:, 0].min(), verts[:,0].max())
		ylim = (verts[:, 1].min(), verts[:,1].max())
		zlim = (verts[:, 2].min(), verts[:,2].max())

		return xlim, ylim, zlim

	def add_legend(self, linestyle, marker, color, label):
		self._ax_legend.plot([], [], linestyle=linestyle, marker=marker,
				color=color, label=label)
		self._ax_legend.legend(ncol=2, mode='expand', fontsize=10)

	def draw(self, ax=None):
		self._fig.canvas.draw()
		self._fig.canvas.flush_events()


def visualize_init_avoid_tube(flock, save=True, fname=None, title=''):
	"""
		For a flock, whose mesh has been precomputed,
		visualize the initial backward avoid tube.
	"""
	# visualize avoid set
	fontdict = {'fontsize':16, 'fontweight':'bold'}

	fig = plt.figure(1, figsize=(16,9), dpi=100)
	ax = plt.subplot(111, projection='3d')
	ax.add_collection3d(flock.mesh)


	xlim = (flock.verts[:, 0].min(), flock.verts[:,0].max())
	ylim = (flock.verts[:, 1].min(), flock.verts[:,1].max())
	zlim = (flock.verts[:, 2].min(), flock.verts[:,2].max())

	# create grid that contains just this zero-level set to avoid computational craze
	gmin = np.asarray([[xlim[0], ylim[0], zlim[0]]]).T
	gmax = np.asarray([[xlim[1], ylim[1], zlim[1]] ]).T

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_zlim(zlim)

	ax.grid('on')
	ax.tick_params(axis='both', which='major', labelsize=10)

	ax.set_xlabel(rf'$x$ (m)', fontdict=fontdict)
	ax.set_ylabel(rf'$z$ (m)', fontdict=fontdict)
	ax.set_zlabel(rf'$\theta$ (rad)',fontdict=fontdict)

	if title:
		ax.set_title(title, fontdict=fontdict)
	else:
		ax.set_title(f'Flock {flock.label}\'s Avoid Tube. Num_agents={flock.N}', fontdict=fontdict)
	ax.view_init(azim=-45, elev=30)

	if save:
		fig.savefig(fname, bbox_inches='tight',facecolor='None')
