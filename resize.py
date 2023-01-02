import os
from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
    folder = "E:/Studio/Magistrale/AV/TestDifFace/training_caip_contest/"
    files = os.listdir(folder)

    for f in tqdm(files):
        img = Image.open(folder + f).resize((512, 512))
        img.save("E:/Studio/Magistrale/AV/TestDifFace/dataset/" + f)