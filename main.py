import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import os.path
from tqdm import tqdm
import math


def load_images():
    img_path = './imgs/'
    imgs = []
    for file in tqdm(os.listdir(img_path)):
        imgs.append(np.array(Image.open(img_path + file)))
    # both input images are from 0-->255
    return imgs


def print_histogram(_histrogram, name, title):
    plt.figure()
    plt.title(title)
    plt.plot(_histrogram, color='#ef476f')
    plt.bar(np.arange(len(_histrogram)), _histrogram, color='#b7b7a4')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.savefig("hist_" + name)


def generate_histogram(img, print, index):
    if len(img.shape) == 3: # img is colorful
        gr_img = np.mean(img, axis=-1)
    else:
        gr_img = img
    '''now we calc grayscale histogram'''
    gr_hist = np.zeros([256])

    for x_pixel in range(gr_img.shape[0]):
        for y_pixel in range(gr_img.shape[1]):
            pixel_value = int(gr_img[x_pixel, y_pixel])
            gr_hist[pixel_value] += 1
    '''normalize Histogram'''
    gr_hist /= (gr_img.shape[0] * gr_img.shape[1])
    if print:
        print_histogram(gr_hist, name="eq_"+str(index), title="Normalized Histogram")
    return gr_hist


def equalize_histogram(histo, L):
    eq_histo = np.zeros_like(histo)
    for i in range(len(histo)):
        eq_histo[i] = int((L - 1) * np.sum(histo[0:i]))
    print_histogram(eq_histo, name="eq_"+str(index), title="Equalized Histogram")


if __name__ == '__main__':
    print("\r\nLoading Images:")
    imgs = load_images()
    print("\r\ngenerating HistogramS:")
    grayscale_histogram_arr = []
    eq_histogram_arr = []
    index = 0
    for img in tqdm(imgs):
        hist_img = generate_histogram(img, print=True, index=index)
        grayscale_histogram_arr.append(hist_img)
        eq_histogram_arr.append(equalize_histogram(hist_img, 8))
        index += 1