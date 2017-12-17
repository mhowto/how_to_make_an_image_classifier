import os
import numpy as np
from PIL import Image


def import_images_from_directory(direcotry, target_size=None, samples=None):
    images = []
    labels = []

    count = 0
    classes = {c: i for i, c in enumerate(os.listdir(direcotry))}
    for c, i in classes.items():
        if samples is not None and count >= samples:
            break
        p = os.path.join(direcotry, c)
        for f in os.listdir(p):
            if samples is not None and count >= samples:
                break
            im = Image.open(os.path.join(p, f))
            if target_size is not None:
                im = resize(im, target_size)
            images.append(np.array(im))
            im.close()
            labels.append(i)
            count += 1
    return np.array(images), labels

def resize(im, target_size):
    return im.resize(target_size, Image.ANTIALIAS)

def shearing(im, ratio):
    pass

def zoom(im, ratio):
    pass

images, labels = import_images_from_directory("data/train", samples=10)
# images = [resize(im, 150, 150) for im in images]
images = images / 255
# [np.array(im)/255 for im in images]
print(images)
labels = np.array(labels)
print(labels)
