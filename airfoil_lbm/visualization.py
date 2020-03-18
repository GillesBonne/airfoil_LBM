import matplotlib.pyplot as plt
import matplotlib.cm
import subprocess
import os


def _plot_field(v, mask=None, title=None):
    plt.clf()
    fig, ax = plt.subplots()

    im = ax.imshow(v.T, cmap=matplotlib.cm.inferno)
    fig.colorbar(im)
    # if not mask is None:
    # ax.matshow(mask, cmap=matplotlib.cm.Blues)


def save_field_as_image(v, mask=None, filename="output"):
    _plot_field(v, mask=mask, title=filename)
    plt.savefig(f"../output/{filename}.png")


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


if __name__ == "__main__":
    make_video()
