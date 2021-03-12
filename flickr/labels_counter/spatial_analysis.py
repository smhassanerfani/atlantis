import os
from scipy import stats
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def images_tensor(imgs_list):
    imgs_tensor = np.zeros((len(imgs_list), 512, 512), dtype=np.uint8)
    for idx, img_path in enumerate(imgs_list):
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        resized_img = cv2.resize(
            image, (512, 512), interpolation=cv2.INTER_NEAREST)
        imgs_tensor[idx] = resized_img
    return imgs_tensor

def label_mode_map(imgs_tensor):

    imgs_mod, _ = stats.mode(imgs_tensor)
    imgs_mod = imgs_mod[0, :, :]
    return imgs_mod


def imshow(imgs_mod, save_path):
    label_name = ['background', 'bicycle', 'boat', 'breakwater', 'bridge', 'building', 'bus', 'canal', 'car', 'cliff', 'culvert',
                  'cypress_tree', 'dam', 'ditch', 'fence', 'fire_hydrant', 'fjord', 'flood', 'glaciers', 'hot_spring', 'lake',
                  'levee', 'lighthouse', 'mangrove', 'marsh', 'motorcycle', 'offshore_platform', 'parking_meter', 'person',
                  'pier', 'pipeline', 'pole', 'puddle', 'rapids', 'reservoir', 'river', 'river_delta', 'road', 'sea', 'ship',
                  'shoreline', 'sidewalk', 'sky', 'snow', 'spillway', 'swimming_pool', 'terrain', 'traffic_sign', 'train', 'truck',
                  'umbrella', 'vegetation', 'wall', 'water_tower', 'water_well', 'waterfall', 'wetland']

    fig, axes = plt.subplots(ncols=1, nrows=1, dpi=150)

    values = np.unique(imgs_mod.ravel())
    selected_labels = [label_name[index] for index in np.unique(imgs_mod)]

    im = axes.imshow(imgs_mod, interpolation='none', cmap=plt.cm.jet)

    # get the colors of the values, according to the colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(
        l=selected_labels[i])) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.08), fancybox=True,
               loc='lower center', ncol=len(values), borderaxespad=0)

    plt.axis('off')
    plt.grid(True)
    # plt.show()
    fig_dir = os.path.join(save_path, "spatial_analysis.png")
    plt.savefig(fig_dir)
    print(f"{fig_dir}")


def colorize_mask(mask, save_path, num_classes=56):
    # palette = [0,0,0,0,64,64,64,64,192,23,46,206,224,128,0,32,64,0,32,64,128,48,141,46,32,128,192,4,125,181,20,56,144,12,51,91,125,209,109,127,196,133,32,
    # 224,128,224,224,0,73,218,110,82,8,150,56,86,248,23,113,200,32,55,235,42,174,171,160,128,96,254,232,244,28,192,189,128,160,160,118,164,54,192,
    # 32,224,192,96,96,221,88,16,169,86,134,96,160,160,92,244,86,190,154,26,58,83,207,160,96,224,138,200,187,32,224,224,80,128,128,118,158,248,254,
    # 54,89,16,128,64,80,128,64,16,64,64,24,139,177,176,0,64,240,64,64,80,32,128,80,160,128,144,96,128,208,96,128,144,32,192,208,32,192,214,141,3,
    # 254,23,113,39,24,186,84,91,147]
    palette = [0,0,0,128,0,0,0,128,0,128,128,0,0,0,128,128,0,128,0,128,128,128,128,128,64,0,0,192,0,0,64,128,0,192,128,0,
    64,0,128,192,0,128,64,128,128,192,128,128,0,64,0,128,64,0,0,192,0,128,192,0,0,64,128,128,64,128,0,192,128,
    128,192,128,64,64,0,192,64,0,64,192,0,192,192,0,64,64,128,192,64,128,64,192,128,192,192,128,0,0,64,128,0,
    64,0,128,64,128,128,64,0,0,192,128,0,192,0,128,192,128,128,192,64,0,64,192,0,64,64,128,64,192,128,64,64,0,
    192,192,0,192,64,128,192,192,128,192,0,64,64,128,64,64,0,192,64,128,192,64,0,64,192,128,64,192,0,192,192,
    128,192,192,64,64,64]
    id_to_colorid = {3:  0,  4:  1,  7:  2,  9:  3, 10:  4, 11:  5, 12:  6, 13:  7, 16:  8, 17:  9,
    18: 10, 19: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 26: 17, 29: 18, 30: 19, 32: 20, 33: 21, 
    34: 22, 35: 23, 36: 24, 38: 25, 39: 26, 40: 27, 42: 28, 43: 29, 44: 30, 45: 31, 53: 32, 54: 33, 
    55: 34, 56: 35,  1: 36,  2: 37,  5: 38,  6: 39, 8: 40, 14: 41, 15: 42, 25: 43, 27: 44, 28: 45, 
    31: 46, 37: 47, 41: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55}
    mask = mask - 1
    mask_copy = np.zeros_like(mask)
    if num_classes==56:
        for k, v in id_to_colorid.items():
            mask_copy[mask == (k-1)] = v
    else:
        mask_copy = mask

    mask = Image.fromarray(mask_copy.astype(np.uint8)).convert('P')
    mask.putpalette(palette)
    mask_path = os.path.join(save_path, "spatial_analysis3.png")
    mask.save(mask_path)

def main(path):
    imgs_list = []
    segID_dir = os.path.join(path, "SegmentationID")
    for root, dirs, imgs in os.walk(segID_dir):
        for img in imgs:
            if img.endswith(".png"):
                imgs_list.append(os.path.join(root, img))

    imgs_tensor = images_tensor(imgs_list)
    imgs_mod = label_mode_map(imgs_tensor)
    colorize_mask(imgs_mod, path)
    # imshow(imgs_mod, path)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
