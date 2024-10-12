"""
EECS 445 - Introduction to Machine Learning
Winter 2024  - Project 2
Visualize Landmarks
    This will open up a window displaying randomly selected training
    images. The label of the image is shown. Click on the figure to
    refresh with a set of new images. You can save the images using
    the save button. Close the window to break out of the loop.

    The success of this script is a good indication that the data flow
    part of this project is running smoothly.

    Usage: python visualize_data.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pandas as pd

from dataset import resize, ImageStandardizer, LandmarksDataset
from imageio.v3 import imread
from utils import config, denormalize_image

OUT_FILENAME = "visualize_data.png"

training_set = LandmarksDataset("train")
training_set.X = resize(training_set.X)
standardizer = ImageStandardizer()
standardizer.fit(training_set.X)

metadata = pd.read_csv(config("csv_file"))
metadata = metadata[metadata["partition"]
                    != "challenge"].reset_index(drop=True)

N = 4
fig, axes = plt.subplots(nrows=2, ncols=N, figsize=(2 * N, 2 * 2))


pad = 3
axes[0, 0].annotate(
    "Original",
    xy=(0, 0.5),
    xytext=(-axes[0, 0].yaxis.labelpad - pad, 0),
    xycoords=axes[0, 0].yaxis.label,
    textcoords="offset points",
    size="large",
    ha="right",
    va="center",
    rotation="vertical",
)
axes[1, 0].annotate(
    "Preprocessed",
    xy=(0, 0.5),
    xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
    xycoords=axes[1, 0].yaxis.label,
    textcoords="offset points",
    size="large",
    ha="right",
    va="center",
    rotation="vertical",
)

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])


def display_imgs():
    rand_idx = np.random.choice(
        np.arange(len(metadata)), size=N, replace=False)
    X, y = [], []
    for idx in rand_idx:
        filename = os.path.join(config("image_path"),
                                metadata.loc[idx, "filename"])
        X.append(imread(filename))
        y.append(metadata.loc[idx, "semantic_label"])

    for i, (xi, yi) in enumerate(zip(X, y)):
        axes[0, i].imshow(xi)
        axes[0, i].set_title(yi.replace("_", "\n"))

    X_ = resize(np.array(X))
    X_ = standardizer.transform(X_)
    for i, (xi, yi) in enumerate(zip(X_, y)):
        axes[1, i].imshow(denormalize_image(xi), interpolation="bicubic")


def on_close(event):
    print("Exiting...")
    exit(0)


fig.canvas.mpl_connect("close_event", on_close)

if platform.system() == "Linux" and "DISPLAY" not in os.environ:
    print(f'No window server found, switching to writing first {N} images to file "{OUT_FILENAME}"')
    display_imgs()
    fig.savefig(OUT_FILENAME, bbox_inches="tight")
    exit(0)

print("Displaying images. Click on the figure to refresh. Close the figure to exit.")

plt.ion()
while True:
    display_imgs()
    plt.draw()
    if plt.get_fignums():
        print(plt.get_fignums())
    else:
        print(None)
    fig.savefig(OUT_FILENAME, bbox_inches="tight")
    if plt.waitforbuttonpress(0) == None:
        break
plt.ioff()

print("Exiting...")
