import numpy as np

n_patches = int(1e6)

alpha = 0
beta = 0


std_t = np.zeros(3)
d_t = np.zeros(3)

for n in range(1, n_patches+1):
    # create the patch
    color = np.random.rand(3)[None, None, :]
    patch = np.ones((3, 3, 3)) * color

    # add noise and clip
    patch1 = patch + np.sqrt(patch * alpha + beta) * np.random.randn(*patch.shape)
    patch1 = np.clip(patch1, 0.0, 1.0)

    patch2 = patch + np.sqrt(patch * alpha + beta) * np.random.randn(*patch.shape)
    patch2 = np.clip(patch2, 0.0, 1.0)

    # compute statistics and store
    curr_std = 0.5 * ( np.std(patch1, axis=(0, 1)) + np.std(patch2, axis=(0, 1)) )
    std_t = (std_t * (n-1) + curr_std) / n

    curr_mean1 = np.mean(patch1, axis=(0, 1))
    curr_mean2 = np.mean(patch2, axis=(0, 1))
    diff_mean = np.abs(curr_mean1 - curr_mean2)
    d_t = (d_t * (n-1) + diff_mean) / n


print('sigma_t = [%2.5f, %2.5f, %2.5f]' % (sigma_t[0], sigma_t[1], sigma_t[2]))
print('d_t =     [%2.5f, %2.5f, %2.5f]' % (d_t[0], d_t[1], d_t[2]))

