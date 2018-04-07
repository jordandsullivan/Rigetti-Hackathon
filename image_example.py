from image_processing import *

path = "6.png"
img = load_image(path)
print("Features (HR, VR) for", path, full_process(img))

