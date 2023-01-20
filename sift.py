import utils
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import cv2


def find_extrema(D_pyramid, threshold):
    point_list = []
    
    N_octave = len(D_pyramid)
    N_pic_per_octave = len(D_pyramid[0])
    
    for n_octave in range(N_octave):
        
        (N_x, N_y) = D_pyramid[n_octave][0, :, :].shape
        
        for n_pic in range(1, N_pic_per_octave - 1):
        
            pic_i = D_pyramid[n_octave][n_pic, :, :]
            pic_i_plus_1 = D_pyramid[n_octave][n_pic + 1, :, :]
            pic_i_minus_1 = D_pyramid[n_octave][n_pic - 1, :, :]
            
            for x in range(1, N_x - 1):
                for y in range(1, N_y - 1):
                    point = pic_i[x, y]
                    
                    if abs(point) >= threshold:
                    
                        neighbour_slice = np.s_[x-1 : x+2, y-1 : y+2]
                        check = np.concatenate([pic_i[neighbour_slice] - point, pic_i_minus_1[neighbour_slice] - point, pic_i_plus_1[neighbour_slice] - point])
                        if (check.max() * check.min() >= 0) and ((check.min() < 0) or (check.max() > 0)):
                            point_list.append((x, y, n_octave))

        
    return point_list


def sift(img, sigma, threshold_extrema):
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    k = 2
    N = 5
    M = 3

    D_pyramid = []
    for m in range(M):
        
        (N_x, N_y) = img.shape
        scaled_img = cv2.resize(img, dsize = (N_y // (2**m), N_x // (2**m)), interpolation=cv2.INTER_CUBIC)
        D_octave = []
        
        for n in range(1, N+1):
            
            try:
                new_filter = gaussian_filter(scaled_img, sigma * (k**n))
                D = new_filter - saved_filter
                saved_filter = new_filter
                
            except:
                saved_filter = gaussian_filter(scaled_img, sigma * k)
                D = saved_filter - gaussian_filter(scaled_img, sigma)

            D_octave.append(np.expand_dims(D, 0))
            
        D_octave = np.concatenate(D_octave, axis = 0)
        print(D_octave.shape)
        D_pyramid.append(D_octave)

    extrema = find_extrema(D_pyramid, threshold_extrema)
        
    return D_pyramid, extrema


import matplotlib.pyplot as plt
picture = utils.load_data()['road1.png']['picture']
G, extrema = sift(picture, 1.6, 1)
list_x = []
for x in extrema:
    if x[2] == 0:
        print(x)

N = 5
M = 3
fig, axs = plt.subplots(M, N)
for i in range(N):
    for j in range(M):
        axs[j, i].imshow(G[j][i])

plt.show()
