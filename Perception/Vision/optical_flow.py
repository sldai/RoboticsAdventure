import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.registration import optical_flow_tvl1
from skimage.util import img_as_float

# Get a VideoCapture object from video and store it in vс
vc = cv2.VideoCapture("Cars.mp4")

def read_one_frame():
    for i in range(2):
        # Read first frame
        _, frame = vc.read()
        # Scale and resize image
        resize_dim = 600
        max_dim = max(frame.shape)
        scale = resize_dim/max_dim
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        yield frame

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_as_float(gray)
    
def upsample_flow(u_prev, v_prev):
    ''' You may implement this method to upsample optical flow from
    previous level
    u_prev, v_prev -> optical flow from prev level
    u, v -> upsampled optical flow to the current level
    '''
    if u_prev is None and v_prev is None:
        return u_prev, v_prev
    u = resize(u_prev,(u_prev.shape[0]*2,u_prev.shape[1]*2),order=1)
    v = resize(v_prev,(u_prev.shape[0]*2,u_prev.shape[1]*2),order=1)
    u = u*2
    v = v*2
    return u, v

def OpticalFlowRefine(im1,im2,window, u_prev=None, v_prev=None):
    '''
    Inputs: the two images at current level and window size
    u_prev, v_prev - previous levels optical flow
    Return u,v - optical flow at current level
    '''
    """ ==========
    YOUR CODE HERE
    ========== """
    im2_warp = np.array(im2)
    
    # force to match shape
    u_prev = resize(u_prev,im1.shape,order=1) 
    v_prev = resize(v_prev,im1.shape,order=1) 
    nr, nc = im2_warp.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                     indexing='ij')
    
    inverse_map = np.array([row_coords+v_prev, col_coords+u_prev])
    for y in range(nr):
        for x in range(nc):
            y_pred = inverse_map[0][y,x]
            x_pred = inverse_map[1][y,x]
            # filling points outside mode "nearest"
            y_pred = np.clip(y_pred,0,nr-1)
            x_pred = np.clip(x_pred,0,nc-1)
            im2_warp[y,x] = im2[int(round(y_pred)),int(round(x_pred))]

    # preform LK using im1, im2_warp
    Ix = convolve(im1,np.array([[1,0,-1]]),mode='nearest')
    Iy = convolve(im1,np.array([[1,0,-1]]).T,mode='nearest')
    It = im2_warp-im1
    IxIx = Ix*Ix
    IxIy = Ix*Iy
    IyIy = Iy*Iy
    IxIt = Ix*It
    IyIt = Iy*It
    sum_kernel = np.ones([window,window])
    IxIx_sum = convolve(IxIx, sum_kernel, mode='nearest')
    IxIy_sum = convolve(IxIy, sum_kernel, mode='nearest')
    IyIy_sum = convolve(IyIy, sum_kernel, mode='nearest')
    IxIt_sum = convolve(IxIt, sum_kernel, mode='nearest')
    IyIt_sum = convolve(IyIt, sum_kernel, mode='nearest')
    
    u = np.zeros_like(u_prev)
    v = np.zeros_like(v_prev)
    for y in range(nr):
        for x in range(nc):
            # y_pred = inverse_map[0][y,x]
            # x_pred = inverse_map[1][y,x]
            # if (0+window//2<=y_pred<=nr-1-window//2) and (0+window//2<=x_pred<=nc-1-window//2):
            A = np.array([[IxIx_sum[y,x], IxIy_sum[y,x]],[IxIy_sum[y,x],IyIy_sum[y,x]]])
            b = -np.array([IxIt_sum[y,x], IyIt_sum[y,x]])
            if np.linalg.det(A) > 1e-4: # nonsingular
                uv = np.linalg.inv(A) @ b
            else: # aperture problem
                uv = np.zeros(2)

            u[y,x] = uv[0]
            v[y,x] = uv[1]
    u = u_prev + u
    v = v_prev + v
    return u, v

def LucasKanadeMultiScale(im1,im2,window, numLevels=2):
    '''
    Implement the multi-resolution Lucas kanade algorithm
    Inputs: the two images, window size and number of levels
    if numLevels = 1, then compute optical flow at only the given image level.
    Returns: u, v - the optical flow
    '''
    
    """ ==========
    YOUR CODE HERE
    ========== """
    im1 = np.array(im1)
    im2 = np.array(im2)

    # first build the pyramid
    pyramid = [[im1,im2]]
    for i in range(1,numLevels):
        im1 = resize(im1, (im1.shape[0]//2, im1.shape[1]//2), anti_aliasing=True)
        im2 = resize(im2, (im2.shape[0]//2, im2.shape[1]//2), anti_aliasing=True)
        pyramid.append([im1,im2])
    u = np.zeros([pyramid[-1][0].shape[0]//2, pyramid[-1][0].shape[1]//2])
    v = np.zeros_like(u)
    for i in range(numLevels-1, -1, -1):
        u,v = upsample_flow(u, v)
        
        im1 = pyramid[i][0]
        im2 = pyramid[i][1]
        u,v = OpticalFlowRefine(im1, im2, window, u, v)
        
    return u, v

def plot_optical_flow(img0,img1,U,V,titleStr, color=False):
    '''
    Plots optical flow given U,V and the images
    '''
    
    # Change t if required, affects the number of arrows
    # t should be between 1 and min(U.shape[0],U.shape[1])
    t=8
    
    # Subsample U and V to get visually pleasing output
    U1 = U[::t,::t]
    V1 = V[::t,::t]
    
    # Create meshgrid of subsampled coordinates
    r, c = img0.shape[0],img0.shape[1]
    cols,rows = np.meshgrid(np.linspace(0,c-1,c), np.linspace(0,r-1,r))
    cols = cols[::t,::t]
    rows = rows[::t,::t]
    
    # Plot optical flow
    plt.figure(figsize=(8,8))
    plt.subplot(211)
    plt.imshow(img0, alpha=0.5)
    plt.imshow(img1, alpha=0.5)
    plt.title('Overlayed Images')
    plt.subplot(212)
    if color:
        plt.imshow(img0)
    else:
        plt.imshow(grayscale(img0), cmap='gray')
    plt.quiver(cols,rows,U1,-V1, scale=1, scale_units='xy')
    plt.title(titleStr)
    plt.show()

def main():
    images = []
    for rgb_f in read_one_frame():
        images.append(rgb_f)
    numLevels=5
    window = 25
    # Plot
    # u,v=LucasKanadeMultiScale(grayscale(images[0]),grayscale(images[1]),window,numLevels)
    v, u = optical_flow_tvl1(grayscale(images[0]), grayscale(images[-1]))
    print(np.max(v))
    plot_optical_flow(images[0],images[-1],u*10,v*10, \
                    'levels = ' + str(numLevels) + ', window = '+str(window))

if __name__ == "__main__":
    main()