from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np

def downsize(img):
    return img.resize((28, 28), Image.ANTIALIAS)

def get_HR(img):
    """Dark pixels in left half / Dark pixels in right half."""
    left_total, right_total = 0, 0
    width, height = img.size
    
    for x in range(width):
        for y in range(height):
            r, g, b, a = img.getpixel((x, y))
            brightness = (r + g + b) / 3
            if x < (width / 2):
                left_total += 1 - (brightness / 255)
            else:
                right_total += 1 - (brightness / 255)
    
    return left_total / right_total

def get_VR(img):
    """Dark pixels in top half / Dark pixels in bottom half."""
    top_total, bottom_total = 0, 0
    width, height = img.size
    
    for x in range(width):
        for y in range(height):
            r, g, b, a = img.getpixel((x, y))
            brightness = (r + g + b) / 3
            if y < (height / 2):
                bottom_total += 1 - (brightness / 255)
            else:
                top_total += 1 - (brightness / 255)
    
    return top_total / bottom_total

def full_process(img):
    """Features HR, VR as an array."""
    img = downsize(img)
    return [get_HR(img), get_VR(img)]

def test_downsize(img_str):
    downsized_image = downsize(Image.open(img_str))
    imshow(np.asarray(downsized_image))

def test_HR(img_str):
    downsized_image = downsize(Image.open(img_str))
    print(get_HR(downsized_image))

def test_VR(img_str):
    downsized_image = downsize(Image.open(img_str))
    print(get_VR(downsized_image))
