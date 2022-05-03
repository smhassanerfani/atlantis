import os
import numpy as np
from PIL import Image
from scipy import stats
from palette import colorize_mask

def get_image_path(path="../dataset/s1a/canal/masks"):
    img_pth = []
    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if file.endswith(".png"):
                img_pth.append(os.path.join(root, file))
    return img_pth


def main():
    image_path_list = get_image_path()
    # print(image_path_list)
    image_tensor = np.zeros((len(image_path_list), 500, 500))
    print(image_tensor.shape)
    for idx, img_pth in enumerate(image_path_list):
        image = Image.open(img_pth)
        image = image.resize((500, 500), resample=Image.NEAREST)

        image_tensor[idx] = np.array(image)

    m = stats.mode(image_tensor)
    m = m[0].squeeze(axis=0)
    img = colorize_mask(m)
    img.show()



if __name__ == "__main__":
    main()