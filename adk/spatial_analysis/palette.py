import json
import numpy as np
from PIL import Image

with open("./labels_info.json", 'r') as jf:
    atlantis = json.load(jf)

palette = []
for label in atlantis:
    palette.append(label["color"][0])
    palette.append(label["color"][1])
    palette.append(label["color"][2])

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    mask_copy = np.zeros_like(mask)
    mask_copy = mask - 1
    new_mask = Image.fromarray(mask_copy.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
