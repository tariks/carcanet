import glob
from pathlib import Path
import urllib.request

import numpy as np
from cv2 import imread
from keras.layers import ReLU
from PIL import Image
from skimage.transform import resize

from .enhance_utils import get_unet, input_shape


unet_path = Path(__file__).parent / 'unet_weights.hdf5'
if not unet_path.exists():
    url = 'https://github.com/tariks/glyptic/raw/refs/heads/main/assets/unet_weights.hdf5'
    urllib.request.urlretrieve(url, unet_path)


def batch(iterable, n=4):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]



def enhance(infiles, outdir):
    model = get_unet(do=0.1, activation=ReLU)
    model.load_weights(str(unet_path))

    for batch_files in batch(infiles, n=4):

        imgs = [
            resize(imread(image_path.resolve()) / 255.0, input_shape)
            for image_path in batch_files
        ]

        imgs = np.array(imgs)
        pred = model.predict(imgs)

        for i, image_path in enumerate(batch_files):

            image_base = image_path.stem

            pred_ = pred[i, :, :, 0]
            image = pred_ * 255
            image = Image.fromarray(image)
            image = image.convert("L")
            outfile = outdir / f"{image_base}.png"
            image.save(outfile, optimize=True)
