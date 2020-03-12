import matplotlib.pyplot as plt
from matplotlib import cm


def save_field_as_image(v, filename="output"):
    plt.clf()
    plt.imshow(v.transpose(), cmap=cm.Blues)
    plt.colorbar()
    plt.savefig(f"output/{filename}.png")


def show_field(v):
    # f, (ax1) = plt.subplots(1, 1, figsize=(16, 6))
    ax = plt.matshow(v.T)
    plt.colorbar()
    # plt.clim(0, 2)
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
