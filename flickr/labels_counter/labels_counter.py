import numpy as np
from skimage import io
import os


def func(path):
    class_array = np.zeros(57)

    for root, dirs, files in os.walk(path):
        for img in files:
            if img.endswith(".png"):
                image = io.imread(os.path.join(root, img))
                imageID = np.unique(image)
                for id in imageID:
                    class_array[id] += 1
    return class_array


def main(func, path, subdir="masks"):
    result = func(os.path.join(path, subdir))
    np.savetxt(f'{path}/class_frequency.csv', result, delimiter=',')


if __name__ == "__main__":
    import sys
    main(func, sys.argv[1])
