import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def draw_edges(vertices, edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for edge in edges:
        v0, v1 = vertices[edge[0]], vertices[edge[1]]
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], 'b')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.show()

def plot_static_path(path, time):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(path[:,0], path[:,1], path[:,2], c=time, alpha = 1, cmap='viridis')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.show()

def plot_dyn_path(path, t_array, skip_frames=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(path[:,0], path[:,1], path[:,2], c=t_array, alpha=1, cmap='viridis', zorder=0)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.grid(False)

    points = []
    max_trail_length = len(path)//5

    def update(frame):
        # Only update for frames that are multiples of skip_frames
        actual_frame = frame * skip_frames
        if actual_frame >= len(path):
            return []
        x, y, z = path[actual_frame]
        new_point, = ax.plot([x], [y], [z], 'ro', markersize=3, alpha=1.0, zorder=10)
        points.append((new_point, 1.0))

        if len(points) > max_trail_length:
            oldest_point, _ = points.pop(0)
            oldest_point.remove()

        for i, (point, _) in enumerate(points):
            new_alpha = 1.0 - (1/max_trail_length * (len(points) - i - 1))
            point.set_alpha(new_alpha)
            # point.set_markersize(10*new_alpha)

        return [p[0] for p in points]

    ani = FuncAnimation(fig, update, frames=(len(t_array) - 1) // skip_frames, blit=False, interval=1)
    plt.show()