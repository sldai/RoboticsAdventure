import numpy as np
from scipy.ndimage.filters import convolve 


################## generate Gaussian difference pyramid #########
################## which is sensitive to blob features  #########
def gaussian_filter(sigma): 
    size = 2*np.ceil(3*sigma)+1 
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1] 
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()

def generate_octave(init_level, s, sigma): 
    octave = [init_level] 
    k = 2**(1/s) 
    kernel = gaussian_filter(k * sigma) 
    for _ in range(s+2): 
        next_level = convolve(octave[-1], kernel) 
        octave.append(next_level) 
    return octave

def generate_gaussian_pyramid(im, num_octave, s, sigma): 
    pyr = [] 
    for _ in range(num_octave): 
        octave = generate_octave(im, s, sigma) 
        pyr.append(octave) 
        im = octave[-3][::2, ::2] 
    return pyr

def generate_DoG_octave(gaussian_octave): 
    octave = [] 
    for i in range(1, len(gaussian_octave)):
        octave.append(gaussian_octave[i] - gaussian_octave[i-1])
    
    return np.concatenate([o[:,:,np.newaxis] for o in octave],axis=2)

def generate_DoG_pyramid(gaussian_pyramid):
    pyr = []
    for gaussian_octave in gaussian_pyramid:
        pyr.append(generate_DoG_octave(gaussian_octave))
    return pyr


#################### Extrema detection #######################

