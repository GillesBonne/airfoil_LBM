import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.axes
import matplotlib.figure
import subprocess
import os

palette: matplotlib.colors.LinearSegmentedColormap
palette = copy.copy(matplotlib.cm.inferno)
palette.set_bad('#666')


def save_field_as_image(v, mask=None, filename="output"):
    """
    Plots a field for a given MxN grid and saves it to a given file
    :param v: Values for a colorplot
    :param fig: (optional) a figure to use. If left empty, it creates a new figure and axis.
    :param ax: (optional) an axis to use. If left empty, it creates a new figure and axis.
    :param mask: (optional) an obstacle mask. This is filled with NaN values, which the colormap can give a different
     color. Only used if v is supplied.
    :param title: (optional) a title for the figure.
    :return:
    """
    _plot_field(v, mask=mask, title=filename)
    plt.savefig(f"../output/{filename}.png", dpi=300)


def save_streamlines_as_image(vx, vy, v=None, mask=None, title=None, filename="output"):
    """
    Shows the streamlines for a supplied vector field and saves it to a given file.
    :param vx: MxN grid of x-component of the vector
    :param vy: MxN grid of y-component of the vector
    :param v: (optional) a background colorplot
    :param fig: (optional) a figure to use. If left empty, it creates a new figure and axis.
    :param ax: (optional) an axis to use. If left empty, it creates a new figure and axis.
    :param mask: (optional) an obstacle mask. This is filled with NaN values, which the colormap can give a different
     color. Only used if v is supplied.
    :param title: (optional) a title for the figure.
    :return:
    """
    _show_streamlines(vx, vy, v, mask=mask, title=title)
    plt.savefig(f"../output/{filename}.png", dpi=300)


def show_field(v, mask=None, title=None):
    _plot_field(v, mask=mask, title=title)
    plt.show()


def plot_crossection(v):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(v[2, :])
    ax2.plot(v[:, 2])
    plt.show()


def plot_2d(y):
    # f, (ax1) = plt.subplots(1, 1, figsize=(16, 6))
    plt.plot(y)
    plt.show()


def make_video(folder='../output/velx', filename='output.mp4'):
    subprocess.run(['ffmpeg', '-i', os.path.join(folder, '%08d.png'),
                    '-framerate', '50', os.path.join(folder, filename)])


def _plot_field(v, fig=None, ax=None, mask=None, title=None):
    """
    Plots a field for a given MxN grid.
    :param v: Values for a colorplot
    :param fig: (optional) a figure to use. If left empty, it creates a new figure and axis.
    :param ax: (optional) an axis to use. If left empty, it creates a new figure and axis.
    :param mask: (optional) an obstacle mask. This is filled with NaN values, which the colormap can give a different
     color. Only used if v is supplied.
    :param title: (optional) a title for the figure.
    :return:
    """
    if fig is None or ax is None:
        plt.clf()
        fig, ax = plt.subplots()

    vprime = v.copy()
    if not mask is None:
        vprime[mask] = np.nan

    im = ax.imshow(vprime.T, cmap=palette)
    fig.colorbar(im)


def _show_streamlines(vx, vy, v=None,
                      fig: matplotlib.figure.Figure = None,
                      ax: matplotlib.axes.Axes = None,
                      mask=None,
                      title=None):
    """
    Shows the streamlines for a supplied vector field.
    :param vx: MxN grid of x-component of the vector
    :param vy: MxN grid of y-component of the vector
    :param v: (optional) a background colorplot
    :param fig: (optional) a figure to use. If left empty, it creates a new figure and axis.
    :param ax: (optional) an axis to use. If left empty, it creates a new figure and axis.
    :param mask: (optional) an obstacle mask. This is filled with NaN values, which the colormap can give a different
     color. Only used if v is supplied.
    :param title: (optional) a title for the figure.
    :return:
    """
    if fig is None or ax is None:
        # plt.figure(dpi=1200)
        plt.clf()
        fig, ax = plt.subplots()

    if v is not None:
        _plot_field(v, fig=fig, ax=ax, mask=mask, title=title)

    xvalues = np.arange(vx.shape[1])
    yvalues = np.arange(vx.shape[0])
    Y, X = np.meshgrid(xvalues, yvalues)

    linewidth = np.ones(v.shape).T * 0.5

    stream_container = ax.streamplot(X.T, Y.T, vx.T, vy.T,
                                     color='#888',
                                     density=(15., 1.),
                                     linewidth=linewidth, arrowsize=0, arrowstyle='-')
    stream_container.lines.set_alpha(0.618 + 0.1)  # Triggered


if __name__ == "__main__":
    make_video()
