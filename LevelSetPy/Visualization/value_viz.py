__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import time, os
import numpy as np
from os.path import join
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from LevelSetPy.Utilities.matlab_utils import *
from LevelSetPy.Grids.create_grid import createGrid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from LevelSetPy.Visualization.mesh_implicit import implicit_mesh
from LevelSetPy.Visualization.settings import buffered_axis_limits

class ValueVisualizer(object):
	def __init__(self, params={}):
		"""
			Use this class to visualize the starting value function's
			zero levelset. If you use a value function different to the
			one specified in the cylinder, you may need to specify the axis
			limits of the plot manually.

			Or send a PR if you can help write a dynamic axes adjuster as the
			level sets' limits varies.

		 Copyright (c) Lekan Molux. https://scriptedonachip.com
		 2021.
		"""
		plt.ion()
		if params.winsize:
			self.winsize=params.winsize
			# create figure 2 for real-time value function updates
			self._fig = plt.figure(2, figsize=self.winsize)
		# mngr = plt.get_current_fig_manager()
		# geom = mngr.window.geometry()
		# x,y,dx,dy = geom.getRect()

		self._gs = gridspec.GridSpec(1, 3, self._fig)
		self._ax_arr = [plt.subplot(self._gs[i], projection='3d') for i in range(2)] + [plt.subplot(self._gs[2])]

		self._labelsize = params.labelsize
		self._init      = False
		self.value      = params.value if isfield(params, 'value') else False
		self._fontdict  = params.fontdict
		self.pause_time = params.pause_time
		self.savedict   = params.savedict
		self.params     = params
		self.color = iter(plt.cm.ocean(np.linspace(0, 1, 3)))
		self._init		= False

		if self.savedict.save and not os.path.exists(self.savedict.savepath):
			os.makedirs(self.savedict.savepath)

		if self._fontdict is None:
			self._fontdict = {'fontsize':12, 'fontweight':'bold'}

		if params.init_conditions or self.value:
			assert isinstance(value, np.ndarray), "value function must be an ndarray."
			self.init_projections(value.ndim)
		else:
			# fig 1 for initial value set
			self._fig_init_val = plt.figure(1, figsize=(16, 9))
			# # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
			# mngr = self._fig_init_val.get_current_fig_manager()
			# geom = mngr.window.geometry()
			# x,y,dx,dy = geom.getRect()
			# mngr.window.setGeometry(100, 200, 640, 580)
			self.viz_init_valueset(params, 0)
			self._fig_init_val.canvas.draw()
			self._fig_init_val.canvas.flush_events()

		self._fig.canvas.draw()
		self._fig.canvas.flush_events()

	def init_projections(self, value_dims, ax_idx = 0):
		"""
		Initialize plots based off the length of the data array.
		"""
		if self.params.init_projections:
			if value_dims==2:
				self._ax_arr[ax_idx] = plt.subplot(self._gs_plots[0])
				self._ax_arr[ax_idx].plot(self.value[0], self.value[1], 'b*') #,markersize=15,linewidth=3, label='Target')
			elif value_dims==3:
				self._ax_arr[ax_idx] = plt.subplot(self._gs_plots[0], projection='3d')
				self._ax_arr[ax_idx].plot(self.value[0], self.value[1], self.value[2], 'b*') #,markersize=15,linewidth=3, label='Target')

			self._ax_arr[ax_idx].set_title('Initial Projections', fontdict=self._fontdict)
			ax_idx += 1
		else:
			value_data = self.params.value_data
			if isfield(self.params, 'grid_bundle') and isbundle(self.params.grid_bundle, self.params):
				g = self.params.grid_bundle
			else: # create the grid
				N = np.asarray(size(value_data)).T
				g = createGrid(np.ones(value_data.ndim, 1), N, N)
			if g.dim != value_data.ndim and g.dim+1 != value_data.ndim:
				raise ValueError('Grid dimension is inconsistent with data dimension!')
			if g.dim == value_data.ndim:
				self.init_valueset(g, value_data, ax_idx)

			self._ax_arr[ax_idx].set_title('Initial Value Set', fontdict=self._fontdict)

		self._ax_arr[ax_idx].xaxis.set_tick_params(labelsize=self._labelsize)
		self._ax_arr[ax_idx].yaxis.set_tick_params(labelsize=self._labelsize)

		self._ax_arr[ax_idx].grid('on')
		self._ax_arr[ax_idx].legend(loc='best', fontdict = self._fontdict)
		self._init = True

	def viz_init_valueset(self, params, ax_idx=0):
		g 		= 	params.grid_bundle
		data 	= 	params.value_data

		self._init		= True

		if g.dim<2:
			# init_ax = self._fig_init_val.gca(projection='3d')
			self._ax_arr[0].plot(g.xs[0],  data, linestyle='-', color=next(self.color))
			self._ax_arr[0].plot(g.xs[0],  np.zeros(size(g.xs[0])), linestyle=':', color='k')
		elif g.dim==2:
			# init_ax = self._fig_init_val.axes(projection='3d')
			self._ax_arr[0].contourf(g.xs[0], g.xs[1], self.value, levels=self.params.level, colors=next(self.color))
			self.title(init_ax, title=f'Initial {self.params.level}-Value Set')# init_ax.set_xlabel('X', fontdict=self.fontdict)

		elif g.dim == 3:
			spacing = tuple(g.dx.flatten().tolist())
			mesh = implicit_mesh(data, level=self.params.level, spacing=spacing,  edge_color='k', face_color='r')
			self.show_3d(g, mesh.mesh, self._ax_arr[0], spacing)

			xlim, ylim, zlim = self.get_lims(mesh)

			self._ax_arr[0].set_xlim(*xlim)
			self._ax_arr[0].set_ylim(*ylim)
			self._ax_arr[0].set_zlim(*zlim)
			self.set_title(self._ax_arr[0], title=f'Starting {self.params.level}-level Value Set')


		elif g.dim == 4:
			# This is useful for the temporal-axis and 3 Cartesian Coordinates
			'Take 6 slice snapshots and show me the 3D projections'
			N=6
			gs = gridspec.GridSpec(2, 3, self._fig_init_val)
			ax =  [plt.subplot(gs[i], projection='3d') for i in range(N)]

			for slice_idx in range(N):
				ax[slice_idx] = plt.subplot(gs[slice_idx], projection='3d')
				xs = g.min[g.dim] + slice_idx/(N+1) * (g.max[g.dim] - g.min[g.dim])
				dim = [0, 0, 0, 1]
				g3D, mesh3D = proj(g, data, dim, xs)
				self.show_3d(g3D, mesh3D, ax[slice_idx], color, spacing)

				self.set_title(ax_idx, f"Projected Slice {g.dim} of Initial Value Function Snapshot {slice_idx}.")

		if self.savedict.save:
			self._fig_init_val.savefig(join(self.savedict.savepath,self.savedict.savename),
								bbox_inches='tight',facecolor='None')

	def show_3d(self, g, mesh, ax_idx, spacing):
		# ax_idx.plot3D(g.xs[0].flatten(), g.xs[1].flatten(), g.xs[2].flatten(), color=next(self.color))
		if isinstance(mesh, list):
			for m in mesh:
				m = implicit_mesh(m, level=self.params.level, spacing=spacing,  edge_color='k', face_color='r')
				ax_idx.add_collection3d(m)
		else:
			ax_idx.add_collection3d(mesh)
		ax_idx.view_init(elev=30., azim=10.)

	def set_title(self, ax, title):
		ax.set_title(title)
		ax.title.set_fontsize(self._fontdict.fontsize)
		ax.title.set_fontweight(self._fontdict.fontweight)

	def add_legend(self, linestyle, marker, color, label):
		self._ax_legend.plot([], [], linestyle=linestyle, marker=marker,
				color=color, label=label)
		self._ax_legend.legend(ncol=2, mode='expand', fontsize=10)

	def viz_value_func(self, gPlot,dataPlot,color,ax):
		"""
			Visualize a surface plot of the entire value function.

			Inputs:
				gPlot: grid on which value function is parameterized
				dataPlot: Value function data defined as an implicit
							function on the grid.
				color: what color to give the value function
				ax: axis on which to graw the value function. If not given, it
				grabs the current axis from pyplot.
			Output:
				Returns the value function 3D plot.
		"""
		if gPlot.dim<2:
			h,  = ax.plot(gPlot.xs[0], np.squeeze(dataPlot), color=color, linewidth=2);
		elif gPlot.dim==2:
			h,  = ax.plot_surface(gPlot.xs[0], gPlot.xs[1], dataPlot, rstride=1, cstride=1,
						cmap='viridis', edgecolor='r', facecolor=color)
		else:
			error('Can not plot in more than 3D!')

		return h

	def levelset_viz(self, g, data, title='', fc='c', ec='k'):
		"""
			Simultaneously visualize the level sets of a value function
			on a 1X3 chart:
				Chart 131: 2D Value function as a surface mesh
				Chart 132: 2D Value function as colored contour levels
				Chart 133: 2D Value zero - set as cyan contour.

			Author: Lekan Molux, October 29, 2021
		"""
		plt.clf()
		self._ax_arr = [plt.subplot(self._gs[i], projection='3d') for i in range(2)] + [plt.subplot(self._gs[2])]

		if g.dim==2:
			self._ax_arr[0].plot_surface(g.xs[0], g.xs[1], data, rstride=1, cstride=1,
									cmap='viridis', edgecolor=ec, facecolor=fc)
			self._ax_arr[0].set_xlabel('X', fontdict=self._fontdict)
			self._ax_arr[0].set_ylabel('Y', fontdict=self._fontdict)
			self._ax_arr[0].set_zlabel('Z', fontdict=self._fontdict)
			self._ax_arr[0].set_title(f'{title}', fontdict=self._fontdict)


			self._ax_arr[1].contourf(g.xs[0], g.xs[1], data, colors=fc)
			self._ax_arr[1].set_xlabel('X', fontdict=self._fontdict)
			self._ax_arr[1].set_title(f'Contours', fontdict=self._fontdict)

			self._ax_arr[2].contour(g.xs[0], g.xs[1], data, levels=0, colors=fc)
			self._ax_arr[2].set_xlabel('X', fontdict=self._fontdict)
			self._ax_arr[2].set_ylabel('Y', fontdict=self._fontdict)
			self._ax_arr[2].grid('on')
			self._ax_arr[2].set_title(f'2D Zero level set', fontdict=self._fontdict)
		elif g.dim == 3:
			# draw the mesh first # see example in test 3d mesh
			# self._ax_arr[0].plot3D(g.xs[0].flatten(), g.xs[1].flatten(), g.xs[2].flatten(), color='cyan')
			'add the zero level set'
			mesh = implicit_mesh(data, level=0., spacing=tuple(g.dx.flatten().tolist()),  edge_color=None, face_color='g')
			self._ax_arr[0].add_collection3d(mesh)
			# self._ax_arr[1].set_xlabel('X', fontdict=self._fontdict)
			# self._ax_arr[1].set_title(f'3D Zero level set', fontdict=self._fontdict)
			self._ax_arr[0].view_init(elev=30., azim=10.)

			xlims, ylims, zlims = self.get_lims(mesh)
			self._ax_arr[0].set_xlim(xlims)
			self._ax_arr[0].set_ylim(ylims)
			self._ax_arr[0].set_zlim(zlims)

			self.set_title(self._ax_arr[1], f'3D Zero level set')

			# 'zero level set with set azimuth and elevation'
			# # mesh = implicit_mesh(data, level=0., spacing=tuple(g.dx.flatten().tolist()),  edge_color=None, face_color='g')
			# # project last dim and visu 2D level set
			xs = 'min' # xs = g.min[g.dim-1] + 3/(N+1) * (g.max[g.dim-1] - g.min[g.dim-1])
			g_red, data_red = proj(g, data, [0, 0, 1], xs)
			self._ax_arr[1].plot_surface(g_red.xs[0], g_red.xs[1], data_red, rstride=1, cstride=1,
						cmap='viridis', edgecolor='k', facecolor='red')
			# self._ax_arr[1].contourf(g_red.xs[0], g_red.xs[1], data_red, colors=next(self.color))
			self.set_title(self._ax_arr[1], f'Value function surface')
			self._ax_arr[1].view_init(elev=30., azim=10.)

			# self._ax_arr[2].contour(g_red.xs[0], g_red.xs[1], data_red, colors=next(self.color))
			self._ax_arr[2].contourf(g_red.xs[0], g_red.xs[1], data_red, colors='blue')
			# # self._ax_arr[2].set_xlabel('X', fontdict=self._fontdict)
			# # self._ax_arr[2].set_title(f'3D Zero level set', fontdict=self._fontdict)
			# self._ax_arr[2].view_init(elev=60., azim=10.)
			self.set_title(self._ax_arr[2], f'Zero level set slice')

		self._fig.tight_layout()
		if self.savedict.save:
			self._fig.savefig(join(self.savedict.savepath,self.savedict.savename),
						bbox_inches='tight',facecolor='None')
		self.draw()
		time.sleep(self.params.pause_time)

	def draw(self):
		for ax in self._ax_arr:
			ax.draw_artist(ax)
		self._fig.canvas.draw()
		self._fig.canvas.flush_events()

	def get_lims(self, mesh):
		xlim = (mesh.verts[:, 0].min(), mesh.verts[:,0].max())
		ylim = (mesh.verts[:, 1].min(), mesh.verts[:,1].max())
		zlim = (mesh.verts[:, 2].min(), mesh.verts[:,2].max())

		return xlim, ylim, zlim
