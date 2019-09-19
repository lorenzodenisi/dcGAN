import os, sys
from PIL import Image

RAW_IMG_PATH = "./raw/"
DATASET_PATH = "./dataset/"
DIM_X = 256
DIM_Y = 256


def main():
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    raw_list = os.listdir(RAW_IMG_PATH)

    for n, raw_img in enumerate(raw_list):
        try:
            img = Image.open(RAW_IMG_PATH+raw_img)
            scaled = img.resize((DIM_X, DIM_Y))
            scaled = scaled.convert('RGB')
            scaled.save(DATASET_PATH + raw_img)
            sys.stdout.write("\r({0}/{1}) {2}".format(n + 1, len(raw_list), raw_img))
        except:
            print("\nError on "+raw_img)

if __name__ == "__main__":
    main()
