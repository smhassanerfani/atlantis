import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def extract_patches(image, image_id, save_path, patch_size=(128, 128)):

    width, height = image.size

    for x in range(0, width, patch_size[0]):
        for y in range(0, height, patch_size[1]):
            box = (x, y, x + patch_size[0], y + patch_size[1])
            patch_name = f"{image_id}.x{x:03d}.y{y:03d}.jpg"
            if np.sum(np.array(image.crop(box)).sum(axis=2) > np.ones((32, 32))) > 1000:
                print(patch_name)
                image.crop(box).save(os.path.join(save_path, patch_name))

labels = ['flood', 'glaciers', 'hot_spring', 'lake', 'pool', 'rapids', 'river', 'sea', 'snow', 'swamp', 'waterfall', 'wetland']
for label in labels:
    for root, dirs, files in os.walk(f"./{label}/", topdown=True):
        for img in files:
            if img.endswith(".jpg"):
                image = Image.open(os.path.join(root, img))
                image_id = img.split(".")[0]
                save_path = os.path.join(root, "patches1")
                try:
                    os.makedirs(save_path)
                except FileExistsError:
                    pass
                extract_patches(image, image_id, save_path)
print("Done!")