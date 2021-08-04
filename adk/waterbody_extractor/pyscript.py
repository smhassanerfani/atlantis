import os
import numpy as np
from skimage import io
import json

with open("labels_ID.json", 'r') as jf:
    atlantis = json.load(jf)


def waterbody_extractor(image, mask, ID):

    if image.ndim == 3:

        img = image.transpose(2, 0, 1)
        mask = (mask == ID)

        img = img * mask

        x, y = np.nonzero(img[0, :, :])

        xl, xr = x.min(), x.max()
        yl, yr = y.min(), y.max()

        cimg = img[:, xl:xr + 1, yl:yr + 1]
        cimg = cimg.transpose(1, 2, 0)
    else:
        img = image
        mask = (mask == ID)

        img = img * mask

        x, y = np.nonzero(img[:, :])

        xl, xr = x.min(), x.max()
        yl, yr = y.min(), y.max()

        cimg = img[xl:xr + 1, yl:yr + 1]

    return cimg


def main(function, rootdir, dataset=atlantis):

    waterbody_path = os.path.join(rootdir, "waterbody")
    try:
        os.makedirs(waterbody_path)
    except FileExistsError as err:
        print(f"FileExistsError: {err}")

    label = rootdir.split("/")[-1]
    ID = atlantis[label]
    images_path = os.path.join(rootdir, "images")
    masks_path = os.path.join(rootdir, "SegmentationID")
    for root, dirs, imgs in os.walk(images_path):
        for img in imgs:
            if img.endswith(".jpg"):
                mask = img.split(".")[0] + ".png"
                image_path = os.path.join(images_path, img)
                mask_path = os.path.join(masks_path, mask)

                image = io.imread(image_path)
                mask = io.imread(mask_path)
                if ID in mask:
                    cimg = function(image, mask, ID)
                    io.imsave(os.path.join(waterbody_path, img), cimg)
                else:
                    print(f"ID: {ID} does not exsit in {img}.")
    print(f"{rootdir} is done!")


if __name__ == "__main__":
    import sys
    main(waterbody_extractor, sys.argv[1])
