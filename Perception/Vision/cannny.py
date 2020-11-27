"""
J. Canny. A computational approach to edge detection. DOI:10.1109/TPAMI.1986.4767851
"""

import numpy as np  
import matplotlib.pyplot as plt
from skimage import data, filters, color


def gaussian_blur(image, sigma):
    """Use Gaussian filter to reduce noise
    """
    return filters.gaussian(image, sigma)

def calc_gradient(image):
    """Calculate image gradient
    """
    dx = filters.sobel(image,axis=1)
    dy = filters.sobel(image,axis=0)
    g_mag = np.hypot(dx, dy)
    g_direct = np.arctan2(dy, dx)
    return g_mag, g_direct

def non_maximal_suppresion(g_mag, g_direct):
    """Compare the pixel gradient magnitude with its neighbor
    in gradient direction, only the local maximal are preserved
    """
    nms = np.zeros_like(g_mag)
    for y in range(1,g_mag.shape[0]-1):
        for x in range(1,g_mag.shape[1]-1):
            if g_mag[y,x]>0:
                theta = g_direct[y, x]
                dx, dy = np.cos(theta), np.sin(theta)
                dx = int(round(dx))
                dy = int(round(dy))
                q = g_mag[y+dy, x+dx]
                r = g_mag[y-dy, x-dx]
                if g_mag[y,x]>q and g_mag[y,x]>r:
                    nms[y,x] = g_mag[y,x]
    return nms

def hysteresis_threshold(image,low,high):
    """Preserve the image area above high and area above low which are connected to high area.

    Step:
        1. add pixels above high to a queue
        2. breadth first search
    """
    
    OPEN = []
    CLOSE = []
    y_inds, x_inds = np.nonzero(image>high)
    for y_ind, x_ind in zip(y_inds, x_inds):
        OPEN.append([y_ind, x_ind])

    acts = [[dy, dx] for dx in range(-1,2) for dy in range(-1,2)]
    acts.remove([0,0])
    while len(OPEN)>0:
        y_ind, x_ind = OPEN.pop(0)
        if (y_ind, x_ind) not in CLOSE:
            CLOSE.append([y_ind, x_ind])
        for dy, dx in acts:
            v = [y_ind+dy, x_ind+dx]
            if 0<=v[0]<image.shape[0] and 0<=v[1]<image.shape[1]:
                if image[v[0],v[1]] > low and v not in CLOSE:
                    OPEN.append(v)
    ht = np.zeros_like(image)
    for y_ind, x_ind in CLOSE:
        ht[y_ind, x_ind] = image[y_ind, x_ind]
    return ht
        
def apply_canny(image, sigma, low, high):
    if image.shape[-1] == 3: # rgb
        image = color.rgb2gray(image)
    smoothed = gaussian_blur(image, sigma=sigma)
    g_mag, g_direct = calc_gradient(smoothed)
    nms = non_maximal_suppresion(g_mag, g_direct)
    ht = hysteresis_threshold(nms, low, high)
    return ht

def main():
    sigma = 1
    low = 0.05
    high = 0.2
    image = data.coins()
    smoothed = gaussian_blur(image, sigma=sigma)
    g_mag, g_direct = calc_gradient(smoothed)
    nms = non_maximal_suppresion(g_mag, g_direct)
    ht = hysteresis_threshold(nms, low, high)

    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('Original image')

    ax[0, 1].imshow(g_mag, cmap='magma')
    ax[0, 1].set_title('Gradient')

    ax[1, 0].imshow(nms, cmap='magma')
    ax[1, 0].set_title('Non maximal suppresion')

    ax[1, 1].imshow(ht>0, cmap='magma')
    ax[1, 1].set_title('Hysteresis threshold')
    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
