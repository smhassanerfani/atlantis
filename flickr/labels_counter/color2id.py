# Writen by Zhenyao 09/03/2020
from skimage import io, data
from collections import namedtuple
import os
import numpy as np
from PIL import Image

Label = namedtuple('Label', ['name', 'id', 'color', ])

labels = [
    # name id  color
    Label('background', 0, (0, 0, 0)),
    Label('bicycle', 1, (0, 64, 64)),
    Label('boat', 2, (64, 64, 192)),
    Label('breakwater', 3, (23, 46, 206)),
    Label('bridge', 4, (224, 128, 0)),
    Label('building', 5, (32, 64, 0)),
    Label('bus', 6, (32, 64, 128)),
    Label('canal', 7, (48, 141, 46)),
    Label('car', 8, (32, 128, 192)),
    Label('cliff', 9, (4, 125, 181)),
    Label('culvert', 10, (20, 56, 144)),
    Label('cypress_tree', 11, (12, 51, 91)),
    Label('dam', 12, (125, 209, 109)),
    Label('ditch', 13, (127, 196, 133)),
    Label('fence', 14, (32, 224, 128)),
    Label('fire_hydrant', 15, (224, 224, 0)),
    Label('fjord', 16, (73, 218, 110)),
    Label('flood', 17, (82, 8, 150)),
    Label('glaciers', 18, (56, 86, 248)),
    Label('hot_spring', 19, (23, 113, 200)),
    Label('lake', 20, (32, 55, 235)),
    Label('levee', 21, (42, 174, 171)),
    Label('lighthouse', 22, (160, 128, 96)),
    Label('mangrove', 23, (254, 232, 244)),
    Label('marsh', 24, (28, 192, 189)),
    Label('motorcycle', 25, (128, 160, 160)),
    Label('offshore_platform', 26, (118, 164, 54)),
    Label('parking_meter', 27, (192, 32, 224)),
    Label('person', 28, (192, 96, 96)),
    Label('pier', 29, (221, 88, 16)),
    Label('pipeline', 30, (169, 86, 134)),
    Label('pole', 31, (96, 160, 160)),
    Label('puddle', 32, (92, 244, 86)),
    Label('rapids', 33, (190, 154, 26)),
    Label('reservoir', 34, (58, 83, 207)),
    Label('river', 35, (160, 96, 224)),
    Label('river_delta', 36, (138, 200, 187)),
    Label('road', 37, (32, 224, 224)),
    Label('sea', 38, (80, 128, 128)),
    Label('ship', 39, (118, 158, 248)),
    Label('shoreline', 40, (254, 54, 89)),
    Label('sidewalk', 41, (16, 128, 64)),
    Label('sky', 42, (80, 128, 64)),
    Label('snow', 43, (16, 64, 64)),
    Label('spillway', 44, (24, 139, 177)),
    Label('swimming_pool', 45, (176, 0, 64)),
    Label('terrain', 46, (240, 64, 64)),
    Label('traffic_sign', 47, (80, 32, 128)),
    Label('train', 48, (80, 160, 128)),
    Label('truck', 49, (144, 96, 128)),
    Label('umbrella', 50, (208, 96, 128)),
    Label('vegetation', 51, (144, 32, 192)),
    Label('wall', 52, (208, 32, 192)),
    Label('water_tower', 53, (214, 141, 3)),
    Label('water_well', 54, (254, 23, 113)),
    Label('waterfall', 55, (39, 24, 186)),
    Label('wetland', 56, (84, 91, 147))
]

color2Id = {label.color: label.id for label in labels}


def function_color2id(im):
    w, h = im.size
    id_map = np.zeros((h, w), dtype=np.uint8)
    # float_im = np.int32(im)
    for x in range(w):
        for y in range(h):
            color = im.getpixel((x, y))
            id_map[y, x] = color2Id[color]
    return id_map


def main(function, path):
    masks_path = os.path.join(path, "masks")
    SegID_path = os.path.join(path, "SegmentationID")
    os.makedirs(SegID_path)

    for root, dirs, imgs in os.walk(masks_path):
        for img in imgs:
            if img.endswith(".png"):
                seg = Image.open(os.path.join(root, img))
                id_map = function_color2id(seg)
                io.imsave(os.path.join(SegID_path, img), id_map)
    print(f"{path} id done!")


if __name__ == "__main__":
    import sys
    main(function_color2id, sys.argv[1])
