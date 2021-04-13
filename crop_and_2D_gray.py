import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
from PIL import Image

out_folder = "2d_array_images/"
num_images = 500
for i in range(num_images):
    file_add = 'raw_data/{}.png'.format(i)
    # img = mpimg.imread(file_add)
    image = Image.open(file_add).convert('L')
    # width, height = image.size
    # print(width, height)
    # Setting the points for cropped image
    left = 20
    top = 15
    right = left + 265
    bottom = top + 265  
    # Cropped image of above dimension
    # (It will not change orginal image)
    image = image.crop((left, top, right, bottom))
    # Shows the image in image viewer
    # image.show()

    # print(image.format)
    # print(image.size)
    # print(image.mode)
    # image.show()
    data = np.asarray(image)
    # plt.matshow(data)
    # plt.show()
    # exit()
    # print(data)
    with open(out_folder + '{}.npy'.format(i), 'wb') as f:
        np.save(f, data)
