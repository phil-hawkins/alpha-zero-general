import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import sys

sys.path.append('..')
from hex.pytorch.graph_hex_board import GraphHexBoard

board = GraphHexBoard.new_vortex_board(6)
vor = Voronoi(board.tri.points, qhull_options="Qbb Qc Qz")
p1 = vor.vertices[vor.regions[20]]
p = Polygon(p1)

plt.rcParams['figure.figsize'] = [10, 10]
fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_colors='blue', line_width=2, line_alpha=0.8, point_size=2)
ax = plt.gca()
plt.triplot(board.tri.points[:, 0], board.tri.points[:, 1], board.tri.simplices)
plt.plot(board.tri.points[:, 0], board.tri.points[:, 1], 'o', picker=20)

ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)


def onpick(event):
    # thisline = event.artist
    # xdata = thisline.get_xdata()
    # ydata = thisline.get_ydata()
    # ind = event.ind
    # points = tuple(zip(xdata[ind], ydata[ind]))
    node_ndx = event.ind[0]
    print('onpick node:', node_ndx)

    #region_ndx = vor.point_region[node_ndx]
    #p1 = vor.vertices[vor.regions[region_ndx]]
    p1 = board.vor_regions[node_ndx]
    p = Polygon(p1)
    ax = plt.gca()
    ax.add_patch(p)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()