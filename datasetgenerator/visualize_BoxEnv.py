from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
import os


from matplotlib.patches import ConnectionPatch
import matplotlib.patches as patches
import matplotlib as mpl
from utils.BoxPlacer import Env
import cv2
import sys

parser = argparse.ArgumentParser(description="Convert Dataset")
parser.add_argument("dataset_path")
parser.add_argument("dataset_item", type=int)
args = parser.parse_args()

data_path = Path(args.dataset_path)

np.set_printoptions(threshold=sys.maxsize)

def annotate_axes(ax, text, fontsize=18):
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="darkgrey",
    )

def grid_rect_patch(ax, pos, size, angle, color):
    origin = np.array(pos) - np.array(size) / 2
    rect = patches.Rectangle(
        (origin[1], origin[0]),
        size[1],
        size[0],
        linewidth=1,
        edgecolor=color,
        facecolor="none",
        fill=True,
    )
    transform = (
        mpl.transforms.Affine2D().rotate_around(pos[1], pos[0], -angle) + ax.transData
    )
    rect.set_transform(transform)
    ax.add_patch(rect)


def visualize(data_id):
    # print(data_id)
    datapoint = {}
    with open(data_path / "meta" / (data_id + ".json"), "r") as file:
        meta = json.load(file)

    
    (data_path / "env_img").mkdir(parents=True, exist_ok=True)

    env = Env(meta["cfg"])
    env.restore_state(meta)
    
    sensor_img_rotations = {"raw": 180, "spherical": 180, "polar": 0}
    sensor_imgs = {"raw": {}, "spherical": {}, "polar": {}}
    for img_type_key, imgs in sensor_imgs.items():
        for sensor_path in sorted(data_path.glob(f"sensor_{img_type_key}_*")):
            img = Image.open(sensor_path / (data_id + ".png")).convert("RGB")
            imgs[sensor_path.stem] = np.array(
                img.rotate(sensor_img_rotations[img_type_key])
            )

    outer = [["upper"] * max(1, len(sensor_imgs["raw"]))]
    for imgs in sensor_imgs.values():
        if len(imgs) > 0:
            outer += [list(imgs.keys())]

    fig, axd = plt.subplot_mosaic(outer, constrained_layout=True, figsize=(12, 6))

  
    env.render(axd["upper"])

    plt.show()
    
    plt.savefig(data_path / "env_img" / f"{data_id}.png")


def pngtobinary(data_id):
    print(data_path / "env_img" / f"{data_id}.png")
    (data_path / "env_img_binary").mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(data_path / "env_img" / f"{data_id}.png"), 2)
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)



if __name__ == "__main__":
    visualize(f"{args.dataset_item:06d}")
    pngtobinary(f"{args.dataset_item:06d}")

