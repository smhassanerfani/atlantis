import os
from scipy import stats
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


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


def main(path):
    imgs_list = []
    segID_dir = os.path.join(path, "SegmentationID")
    for root, dirs, imgs in os.walk(segID_dir):
        for img in imgs:
            if img.endswith(".png"):
                imgs_list.append(os.path.join(root, img))

    imgs_tensor = images_tensor(imgs_list)
    imgs_mod = label_mode_map(imgs_tensor)
    imshow(imgs_mod, path)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
