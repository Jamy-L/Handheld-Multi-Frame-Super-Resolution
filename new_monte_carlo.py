import numpy as np
import tqdm
import os

n_patches = int(1e4)

iso = 100.0
iso = iso / 100

alpha = 1.80710882e-4  # from measurements
beta = 3.1937599182128e-6 # from https://www.photonstophotos.net/Charts/RN_ADU.htm chart

n_colors = 3
n_brightness_levels = 100

print('##### MC parameters #####')
print('    alpha   = %f' % alpha)
print('    beta    = %f' % beta)
print('    ISO     = %d' % (iso * 100))
print('    #points = %d' % n_brightness_levels)
print('#########################')
print()

std_t = np.zeros(n_brightness_levels+1)
d_t = np.zeros(n_brightness_levels+1)

    
for b in tqdm.tqdm(range(n_brightness_levels + 1)):
    color = b / n_brightness_levels
    for n in range(1, n_patches+1):
        # create the patch
        patch = np.ones((3, 3, n_colors)) * color

        # add noise and clip
        patch1 = patch + iso**2 * np.sqrt(patch/iso * alpha + beta) * np.random.normal(0, scale=1, size=patch.shape)
        patch1 = np.clip(patch1, 0.0, 1.0)

        patch2 = patch + iso**2 * np.sqrt(patch/iso * alpha + beta) * np.random.normal(0, scale=1, size=patch.shape)
        patch2 = np.clip(patch2, 0.0, 1.0)
 
        # compute statistics and store
        # std is computed for each channel.
        # std = (std_R^2 + std_G^2 + std_B^2) ^ 0.5
        # The 2 patches are used to improve sample efficiency
        curr_std = 0.5 * (np.sqrt(np.sum(np.std(patch1, axis=(0, 1))**2)) + np.sqrt(np.sum(np.std(patch2, axis=(0, 1))**2)))
        std_t[b] += curr_std

        curr_mean1 = np.mean(patch1, axis=(0, 1))
        curr_mean2 = np.mean(patch2, axis=(0, 1))
        # color distance
        diff_mean = np.linalg.norm(curr_mean1 - curr_mean2)
        d_t[b] += diff_mean

std_t /= n
d_t /= n


print('sigma_t', std_t)
print('d_t', d_t)

os.makedirs('./data', exist_ok=True)
# np.save('./data/noise_model_std_ISO_%d.npy' % (iso*100), std_t)
# np.save('./data/noise_model_diff_ISO_%d.npy'% (iso*100), d_t)
