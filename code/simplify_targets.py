# remove color from target images. output black and white lines and resize to (400, 288)

import sys
from PIL import Image
# from PIL.ImageOps import grayscale, autocontrast, invert
# from PIL.ImageFilter import FIND_EDGES

output_shape = (288, 400)
infile = sys.argv[1]
outfile = infile.replace(".jpg", "_small.png")

with Image.open(infile) as image:
    gray = image.resize(output_shape)
    # gray = grayscale(gray)
    # gray = autocontrast(gray)
    # gray = gray.resize(output_shape)
    # gray = gray.filter(FIND_EDGES)
    # gray = gray.point(lambda x: 0 if x < 200 else 255, '1')
    # gray.convert('1')
    # gray = invert(gray)
    gray.save(outfile)
