

"""
Ref:
https://github.com/scikit-image/scikit-image/tree/master/skimage/feature
"""
from scipy.ndimage import maximum_filter
import numpy as np  
import matplotlib.pyplot as plt
from skimage import data, filters, color
from itertools import combinations_with_replacement

def _compute_derivatives(image, mode='constant', cval=0):
    """Compute derivatives in axis directions using the Sobel operator.
    Parameters
    ----------
    image : ndarray
        Input image.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    Returns
    -------
    derivatives : list of ndarray
        Derivatives in each axis direction.
    """

    derivatives = [filters.sobel(image, axis=i, mode=mode, cval=cval)
                   for i in range(image.ndim)]

    return derivatives


def structure_tensor(image, sigma=1, mode='constant', cval=0):
    """Compute structure tensor using sum of squared differences.
    The (2-dimensional) structure tensor A is defined as::
        A = [Arr Arc]
            [Arc Acc]
    which is approximated by the weighted sum of squared differences in a local
    window around each pixel in the image. This formula can be extended to a
    larger number of dimensions (see [1]_).
    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used as a
        weighting function for the local summation of squared differences.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    """
    derivatives = _compute_derivatives(image, mode=mode, cval=cval)

    # structure tensor
    A_elems = [filters.gaussian(der0 * der1, sigma, mode=mode, cval=cval)
               for der0, der1 in combinations_with_replacement(derivatives, 2)]
    return A_elems

def gaussian_blur(image, sigma):
    """Use Gaussian filter to reduce noise
    """
    return filters.gaussian(image, sigma)


def corner_harris(image, sigma=1, k=0.05):
    """Calculate Harris response of the image
    """
    Arr, Arc, Acc = structure_tensor(image, sigma=sigma, mode='nearest')
    # determinant
    detA = Arr * Acc - Arc ** 2
    # trace
    traceA = Arr + Acc

    response = detA - k * traceA ** 2
    return response 

def corner_shi_tomasi(image, sigma=1):
    """Compute Shi-Tomasi (Kanade-Tomasi) corner measure response image.
    """

    Arr, Arc, Acc = structure_tensor(image, sigma, mode='nearest')

    # minimum eigenvalue of A
    response = ((Arr + Acc) - np.sqrt((Arr - Acc) ** 2 + 4 * Arc ** 2)) / 2

    return response

def corner_kitchen_rosenfeld(image, mode='constant', cval=0):
    """Compute Kitchen and Rosenfeld corner measure response image.
    The corner measure is calculated as follows::
        (imxx * imy**2 + imyy * imx**2 - 2 * imxy * imx * imy)
            / (imx**2 + imy**2)
    Where imx and imy are the first and imxx, imxy, imyy the second
    derivatives.
    """

    imy, imx = _compute_derivatives(image, mode=mode, cval=cval)
    imxy, imxx = _compute_derivatives(imx, mode=mode, cval=cval)
    imyy, imyx = _compute_derivatives(imy, mode=mode, cval=cval)

    numerator = (imxx * imy ** 2 + imyy * imx ** 2 - 2 * imxy * imx * imy)
    denominator = (imx ** 2 + imy ** 2)

    response = np.zeros_like(image, dtype=np.double)

    mask = denominator != 0
    response[mask] = numerator[mask] / denominator[mask]

    return response


def main():
    from matplotlib import pyplot as plt

    from skimage import data
    from skimage.transform import warp, AffineTransform
    from skimage.draw import ellipse

    # Sheared checkerboard
    tform = AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7,
                            translation=(110, 30))
    image = warp(data.checkerboard()[:90, :90], tform.inverse,
                output_shape=(200, 310))
    # Ellipse
    rr, cc = ellipse(160, 175, 10, 100)
    image[rr, cc] = 1
    # Two squares
    image[30:80, 200:250] = 1
    image[80:130, 250:300] = 1
    image = plt.imread('price_center20.jpeg')
    image = color.rgb2gray(image)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    ax[0,0].imshow(image, cmap='gray')
    ax[0,0].set_title('Original image')

    ax[0,1].imshow(corner_kitchen_rosenfeld(image), cmap='magma')
    ax[0,1].set_title('Kitchen')

    ax[1,0].imshow(corner_harris(image), cmap='magma')
    ax[1,0].set_title('Harris')

    ax[1,1].imshow(corner_shi_tomasi(image), cmap='magma')
    ax[1,1].set_title('Shi-Tomasi')
    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
