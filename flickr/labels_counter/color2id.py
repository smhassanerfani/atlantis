# Writen by Zhenyao 09/03/2020
from skimage import io
from collections import namedtuple
import os
import numpy as np
from PIL import Image

Label = namedtuple('Label', ['name', 'id', 'color', ])

atlantis = [
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


def csv_writer(path, input_list, fields=["label", "pixels", "counter"]):
    import csv
    csv_path = os.path.join(path, "labels_stat.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=fields)

        csv_writer.writeheader()
        for item in input_list:
            csv_writer.writerow(item)


def fun_color2id(img, labels=atlantis):

    color2id = {label.color: label.id for label in labels}
    num_of_classes = len(labels)

    w, h = img.size
    id_map = np.zeros((h, w), dtype=np.uint8)
    pixels_list = np.zeros(num_of_classes, dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            color = img.getpixel((x, y))
            id_map[y, x] = color2id[color]
            pixels_list[color2id[color]] += 1
    return id_map, pixels_list


def main(function, path, labels=atlantis):

    # labels_list = [label.name for label in labels]

    masks_path = os.path.join(path, "masks")
    SegID_path = os.path.join(path, "SegmentationID")
    try:
        os.makedirs(SegID_path)
    except FileExistsError as err:
        print(f"FileExistsError: {err}")

    for root, dirs, imgs in os.walk(masks_path):

        total_pixels = np.zeros(len(labels))
        total_labels_array = np.zeros(len(labels))
        for img in imgs:
            if img.endswith(".png"):
                img_dict = {}
                mask = Image.open(os.path.join(root, img))
                id_map, pixels_list = fun_color2id(mask)
                total_pixels += pixels_list
                io.imsave(os.path.join(SegID_path, img), id_map)
                for unq_id in np.unique(id_map):
                    total_labels_array[unq_id] += 1

    dir_pixels = []
    for label in labels:
        img_dict = {}
        img_dict["label"] = label.name
        img_dict["pixels"] = total_pixels[label.id]
        img_dict["counter"] = total_labels_array[label.id]
        dir_pixels.append(img_dict)

    csv_writer(path, dir_pixels)
    print(f"{path} id done!")


if __name__ == "__main__":
    import sys
    main(fun_color2id, sys.argv[1])
