# %%
"""
#Image Denoising with the FFT Exercise

This code solves the image denoising with FFT exercise.
"""

# %%
pip install -q graphlearning

# %%
import matplotlib.pyplot as plt
import graphlearning as gl

img = gl.datasets.load_image('chairtoy')
img = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2] #convert to grayscale
plt.imshow(img,cmap='gray')

#Check data range and shape
print('Pixel intensity range: (%f,%f)'%(img.min(),img.max()))
print(img.shape)

# %%
"""
Let's now add some noise to the image.
"""

# %%
import numpy as np

f_noisy = img + 0.1*np.random.randn(img.shape[0],img.shape[1])

plt.figure(figsize=(10,10))
plt.imshow(f_noisy,cmap='gray',vmin=0,vmax=1)

# %%
"""
Let's define the functions for Tikhonov and Total Variation Denoising.
"""

# %%
from scipy.fft import ifft2
from scipy.fft import fft2

def even_ext(f):
    """Even extension of an image f

    Args:
        f: Size nxm numpy array for image

    Returns:
        Numpy array of length 2(n-1)x2(m-1) containing even extension
    """
    g = np.hstack((f,f[:,-1:1:-1]))
    return np.vstack((g,g[-1:1:-1,:]))

def tikhonov_denoising(f,lam):
    """Tikhonov regularized image denoising

    Args:
        f: Noisy image (numpy array)
        lam: Regularization parameter

    Returns:
        Denoised image
    """

    fn = even_ext(f)
    n = fn.shape[0]
    k1 = np.zeros((n,n), dtype=int)+np.arange(n, dtype=int)
    k2 = k1.T
    G = 1/(1 + 4*lam - 2*lam*(np.cos(2*np.pi*k1/n) + np.cos(2*np.pi*k2/n)))
    fd = ifft2(G*fft2(fn)).real
    return fd[:f.shape[0],:f.shape[1]]

plt.figure(figsize=(10,10))
plt.imshow(even_ext(img),cmap='gray',vmin=0,vmax=1)
plt.title('Even Extension of an Image')

# %%
"""
Let's now run an experiment comparing Tikhonov to TV denoising.
"""

# %%
f_tik = tikhonov_denoising(f_noisy,2)

plt.figure(figsize=(20,30))
plt.imshow(np.hstack((f_noisy,f_tik)),cmap='gray',vmin=0,vmax=1)