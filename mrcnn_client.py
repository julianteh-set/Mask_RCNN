import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, BooleanVar, Checkbutton

window = tk.Tk()

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

input_dir = ROOT_DIR
output_dir = ROOT_DIR
input_field = tk.Label(text=input_dir)
output_field = tk.Label(text=output_dir)

view_checked = tk.BooleanVar()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def set_input_dir():
    global input_dir

    currdir = os.getcwd()

    folder_path = filedialog.askdirectory(parent=window,
                                          initialdir=currdir,
                                          title='Please select a directory')

    if len(folder_path) > 0:
        if folder_path is None:
            return

        input_dir = folder_path
        input_field["text"] = input_dir


def set_output_dir():
    global output_dir

    currdir = os.getcwd()

    folder_path = filedialog.askdirectory(parent=window,
                                          initialdir=currdir,
                                          title='Please select a directory')

    if len(folder_path) > 0:
        if folder_path is None:
            return

        output_dir = folder_path
        output_field["text"] = output_dir


def process():
    global input_dir
    global output_dir

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)

            if not os.path.exists(file_path) or not file_path.lower().endswith(
                ('.png', '.jpg', '.jpeg')):
                continue

            image = skimage.io.imread(file_path)

            # Run detection
            results = model.detect([image], verbose=1)

            # Visualize results
            r = results[0]

            masked, ax, masked_image = visualize.mask_instances(
                image,
                r['rois'],
                r['masks'],
                r['class_ids'],
                class_names,
                r['scores'],
                show_bbox=False,
                filter=["car", "truck", "bus", "motorcycle", "person"])

            if masked:
                if view_checked.get():
                    visualize.display_masked_image(ax, masked_image)

                # Write out the images to output_dir
                skimage.io.imsave(os.path.join(output_dir, "masked_" + file),
                                  masked_image)
            else:
                print("No mask found, saving original image")

            skimage.io.imsave(os.path.join(output_dir, file), image)


def main():
    global input_dir
    global output_dir
    global input_field
    global output_field

    input_label = tk.Label(text="Input Directory")
    input_label.pack()
    input_button = tk.Button(text="Browse",
                             width=8,
                             height=1,
                             bg="white",
                             fg="black",
                             command=set_input_dir)
    input_field.pack()
    input_button.pack()

    output_label = tk.Label(text="Output Directory")
    output_label.pack()
    output_button = tk.Button(text="Browse",
                              width=8,
                              height=1,
                              bg="white",
                              fg="black",
                              command=set_output_dir)
    output_field.pack()
    output_button.pack()

    view_button = Checkbutton(window,
                              text="View images",
                              variable=view_checked)
    view_button.pack()

    process_button = tk.Button(text="Process Images",
                               width=15,
                               height=3,
                               command=process)
    process_button.pack()

    window.mainloop()


if __name__ == "__main__":
    main()
