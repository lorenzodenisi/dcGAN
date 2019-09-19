import GAN_nn as GAN
import os, sys
from PIL import Image
import numpy as np

import tensorflow as tf

IMG_PATH = "./dataset/"


def main():

    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())

    nn = GAN.DCGAN_large_img((256, 256, 3), 100, (256, 256, 3), "./model")
    dataset = load_data(IMG_PATH)

    nn.train(dataset, 16, 1000)
    nn.generate_collage(5, 3,  256)


def load_data(path):
    imgs = os.listdir(path)

    dataset = []

    print("Loading dataset, " + str(len(imgs)) + " images found")

    for n, img in enumerate(imgs):
        if n < 500:
            sys.stdout.write("\r({0}/{1})".format(n + 1, len(imgs)))
            image = Image.open(path + img)
            dataset.append(np.array(image.getdata()))

    print("\n")
    return dataset


if __name__ == "__main__":
    main()
