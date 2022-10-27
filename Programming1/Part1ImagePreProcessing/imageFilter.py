import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def DoG_filter(size,sigma=1):
    g = gaussian_filter(size,sigma)
    gx = np.zeros(shape = (size,size))
    gy = np.zeros(shape = (size,size))

    for row in range(size):
        x = row - (size // 2)
        for col in range(size): 
            y = col - (size // 2)
            gx[row,col] = -1 * x * g[row,col] / (sigma ** 2)
            gy[row,col] = -1 * y * g[row,col] / (sigma ** 2)
       
    return (gx,gy)

    for row in range(size):
        x = row - (size // 2)
        for col in range(size): 
            y = col - (size // 2)
            result[row,col] = (1 / (np.sqrt(2 * np.pi) * sigma) ) * np.e ** ( (-1 * ( x**2 + y**2)) / (2 * sigma**2)) 

    result *= 1/result.max()

    return result

def gaussian_filter(size,sigma=1):
    g = np.zeros(shape = (size,size))
    for row in range(size):
        x = row - (size // 2)
        for col in range(size): 
            y = col - (size // 2)
            g[row,col] = (1 / (np.sqrt(2 * np.pi) * sigma) ) * np.e ** ( (-1 * ( x**2 + y**2)) / (2 * sigma**2)) 

    g *= 1/g.sum()

    return g

def sobel_convolution(image,gx,gy):
    image_row,image_col = image.shape

    gx_image = convolution(image,gx)
    gy_image = convolution(image,gy)

    filtered_image = np.zeros(shape=(image_row,image_col))
    for row in range(image_row):
        for col in range(image_col):
            filtered_image[row, col] = np.sqrt(gx_image[row,col]**2 + gy_image[row,col]**2)
    
    return filtered_image


def convolution(image,filter):
    image_row,image_col = image.shape
    filter_row,filter_col = filter.shape

    pad_row = int((filter_row - 1) / 2)
    pad_col = int((filter_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_row), image_col + (2 * pad_col)))
    padded_image[pad_row:padded_image.shape[0] - pad_row, pad_col:padded_image.shape[1] - pad_col] = image

    filtered_image = np.zeros(shape=(image_row,image_col))

    for row in range(image_row):
        for col in range(image_col):
            filtered_image[row, col] = np.sum(filter * padded_image[row:row + filter_row, col:col + filter_col])

    return filtered_image

def show_image(image,title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    g3 = gaussian_filter(3)
    g5 = gaussian_filter(5)
    gx,gy = DoG_filter(3)
    for fname in ["filter1_img.jpg","filter2_img.jpg"]:
        iname = fname.split('_')[0]
        ipath = projdir.replace('\\', '/') + '/data/' + fname
        image = np.array(Image.open(ipath).convert('L'))
        show_image(image, iname + ":orginal")

        g3_image = convolution(image,g3)
        show_image(g3_image,iname + ":3x3 Gaussian Filter")

        g5_image = convolution(image,g5)
        show_image(g5_image,iname + ":5x5 Gaussian Filter")

        gx_image = convolution(image,gx)
        show_image(gx_image,iname + ":DoG Filter Horizontal Edge Detector")

        gy_image = convolution(image,gy)
        show_image(gy_image,iname + ":DoG Filter Vertical Edge Detector")

        sobel_image = sobel_convolution(image,gx,gy)
        show_image(sobel_image,iname + ":Sobel Filter")

    print(projdir)