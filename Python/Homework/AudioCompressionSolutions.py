# %%
"""
#PCA-based Audio Compression Solutions

Let's download an audio file. You can use anything you like; there is a file with classical music availble on the course website.
"""

# %%
import urllib.request

urllib.request.urlretrieve('https://github.com/jwcalder/LAML/raw/refs/heads/main/Python/Data/classical.mp3','classical.mp3')

# %%
"""
Let's convert the mp3 to wav and load into Python.
"""

# %%
pip install ffmpeg-python

# %%
from scipy.io import wavfile
import ffmpeg

ffmpeg.input('classical.mp3').output('classical.wav').run()
fs, data = wavfile.read('classical.wav')
data = data.T
print(fs)
data.shape

# %%
"""
Let's also play the audio file.
"""

# %%
from IPython.display import Audio

Audio(data,rate=fs,autoplay=True)

# %%
"""
As in the case of image compression, we'll use the image_to_patches and patches_to_image functions to convert the audio signal into short segments.
"""

# %%
import numpy as np

def image_to_patches(I,patch_size=(16,16)):
    """"
    Converts an image into an array of patches

    Args:
        I: Image as numpy array
        patch_size: tuple giving size of patches to use

    Returns:
        Numpy array of size num_patches x num_pixels_per_patch
    """

    #Compute number of patches and enlarge image if necessary
    num_patches = (np.ceil(np.array(I.shape)/np.array(patch_size))).astype(int)
    image_size = num_patches*patch_size
    J = np.zeros(tuple(image_size.astype(int)))
    J[:I.shape[0],:I.shape[1]]=I

    patches = np.zeros((num_patches[0]*num_patches[1],patch_size[0]*patch_size[1]))
    p = 0
    for i in range(int(num_patches[0])):
        for j in range(int(num_patches[1])):
            patches[p,:] = J[patch_size[0]*i:patch_size[0]*(i+1),patch_size[1]*j:patch_size[1]*(j+1)].flatten()
            p+=1

    return patches

def patches_to_image(patches,image_shape,patch_size=(16,16)):
    """"
    Converts an array of patches back into an image

    Args:
        patches: Array of patches, same as output of image_to_patches
        image_shape: tuple giving the size of the image to return (e.g. I.shape)
        patch_size: tuple giving size of patches

    Returns:
        Image as a numpy array
    """

    #Compute number of patches and enlarge image if necessary
    num_patches = (np.ceil(np.array(image_shape)/np.array(patch_size))).astype(int)
    image_size = num_patches*np.array(patch_size)

    I = np.zeros(tuple(image_size.astype(int)))
    p = 0
    for i in range(num_patches[0]):
        for j in range(num_patches[1]):
            I[patch_size[0]*i:patch_size[0]*(i+1),patch_size[1]*j:patch_size[1]*(j+1)] = np.reshape(patches[p,:],patch_size)
            p+=1

    return I[:image_shape[0],:image_shape[1]]

# %%
"""
Let's convert the audio file into an array of patches. Use a patch size of (n,2), since there are only 2 channels. You can vary n, a lenght 100 or so patch size gives good results.
"""

# %%
psize = (2,64)
X = image_to_patches(data,patch_size=psize)

print(X.shape)

# %%
"""
Let's run PCA on the patches. Let the number of components be a variable that you can play around with.
"""

# %%
from scipy.sparse import linalg
import numpy as np

#Number of principal components to use
num_comps = 16

#Compute the principal components
Vals, V = linalg.eigs(X.T@X,k=num_comps,which='LM')
V = V.real

#Compress the audio by projecting to the linear subspace spanned by V
X_compressed = X@V

#Compute size of compressed audio and compression ratio
compressed_size = X_compressed.size + V.size
comp_ratio = data.size/compressed_size
print('Compression ratio: %.1f:1'%comp_ratio)

# %%
"""
Let's now plot some of the principal components. They look suspiciously like sinusoids!
"""

# %%
import matplotlib.pyplot as plt

for i in range(7):
    plt.plot(np.arange(psize[1]),V[:psize[1],i])

# %%
"""
Let's now decompress the audio by changing coordinates back to the standard ones.
"""

# %%
import numpy as np

#Decompress audio
X_decompressed = X_compressed@V.T
data_decompressed = patches_to_image(X_decompressed,data.shape,patch_size=psize)

# %%
"""
It is difficult to measure the accuracy of the reconstructed audio file, since the real test is how it sounds to the human ear, and two audio signals can sound the same while appearing to be vastly different signals (since phase shifting a pure tone does not change the sound).

So let's play the decompressed audio. How does it sound? What is the highest compression ratio you can achieve without audible noise?
"""

# %%
from IPython.display import Audio

Audio(data_decompressed,rate=fs,autoplay=True)

# %%
"""
To improve the compression, use a windowing function on the patches, and use overlapping patches, as described in the project description.
"""

# %%
from scipy.sparse import linalg
import numpy as np

#Patch size and shift signal
psize = (2,256)
N = int(psize[1]/2)
datashift = data[:,N:]

#Window function
W = np.sin(np.pi*(np.arange(2*N)+0.5)/(2*N))**2
W = np.vstack((W,W)).flatten() #For the two channels

#Convert to paches and multiply by window function
X = image_to_patches(data,patch_size=psize)
Xshift = image_to_patches(datashift,patch_size=psize)

#Number of principal components
num_comps = int(psize[1]/8)

#Compute the principal components
Vals, V = linalg.eigs(X.T@X,k=num_comps,which='LM')
V = V.real

#Compress the audio by projecting to the linear subspace spanned by V
X_compressed = X@V
Xshift_compressed = Xshift@V

#Compute size of compressed audio and compression ratio
compressed_size = X_compressed.size + Xshift_compressed.size + V.size
comp_ratio = data.size/compressed_size
print('Compression ratio: %.1f:1'%comp_ratio)

#Decompress audio and multiply by window function again
X_decompressed = X_compressed@V.T*W
Xshift_decompressed = Xshift_compressed@V.T*W
data_decompressed = patches_to_image(X_decompressed,data.shape,patch_size=psize)
datashift_decompressed = patches_to_image(Xshift_decompressed,datashift.shape,patch_size=psize)

#Add data and datashit
data_decompressed = data_decompressed[:,N:] + datashift_decompressed

# %%
"""
The noise problem should be mitigated to some degree, but you may still hear noise. Audio compression is a much different problem than image compression, and advanced audio compression algorithms use psychoacoustic modeling to discard only the information that is not audible to the human ear.
"""

# %%
from IPython.display import Audio

Audio(data_decompressed,rate=fs,autoplay=True)