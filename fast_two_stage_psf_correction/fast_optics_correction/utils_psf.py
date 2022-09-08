import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.ndimage import center_of_mass


##### Inspired by Mathias Bauer's code from his ICCP18 paper


class PSF(object):
    def __init__(self, psf_path, filter_size=31):
        self.psf_name = psf_path[:-4].split('/')[-1]
        self.locations, self.exposures, self.kernels = self.read_psf_(psf_path, filter_size)
        self.N = len(self.locations)
        self.W = 8712  # Canon EOS 5DSR sensor size
        self.H = 5808  # Canon EOS 5DSR sensor size
        self.filter_size = filter_size

    def __str__(self):
        return self.psf_name + " (%d %dx%d filters)" % (self.N, self.kernels.shape[1], self.kernels.shape[2])

    def __len__(self):
        return self.N

    def get_sensor_size(self):
        return self.H, self.W

    def get_psf_by_index(self, index):
        index = index % self.N
        location = self.locations[index]
        exposure = self.exposures[index]
        kernel = self.kernels[index]
        return location, exposure, kernel

    def get_psf_by_location(self, new_location):
        distances = np.linalg.norm(self.locations - np.array(new_location), axis=1)
        index = np.argmin(distances)
        location = self.locations[index]
        exposure = self.exposures[index]
        kernel = self.kernels[index]
        return location, exposure, kernel

    def get_polar_location(self, location):
        i, j = location
        H, W = self.H, self.W
        i -= (H-1)/2
        j -= (W-1)/2
        r = np.sqrt(i**2 + j**2)
        theta = np.arctan2(-i, j)
        return r, theta

    def read_psf_(self, psf_path, filter_size=31):
        C = loadmat(psf_path)['C']
        N = C.shape[0]  # number of local kernels for this PSF
        X = np.array([C[n, 0][0][0] for n in range(N)]).astype(np.int64)
        Y = np.array([C[n, 1][0][0] for n in range(N)]).astype(np.int64)
        locations = np.stack([Y, X], axis=-1)
        exposures = np.array([C[n, 2][0][0] for n in range(N)]).astype(np.float32)
        exposures = 2**(0.33 * exposures)
        kernels = np.array([C[n, 3] for n in range(N)]).astype(np.float32)
        half_size = filter_size // 2
        central_pix = kernels.shape[1] // 2
        kernels = kernels[:, central_pix - half_size:central_pix + half_size + 1,
                          central_pix - half_size:central_pix + half_size + 1]
        kernels /= (kernels.sum(axis=(1, 2), keepdims=True) + 1e-6)
        return locations, exposures, kernels

    def gaussian_filter_(self, SIGMA, shift=np.array([0.0, 0.0]), k_size=np.array([15, 15])):
        """"
        # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
        # Kai Zhang
        """
        # Set INVCOV matrix using Lambdas and Theta
        INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

        # Set expectation position
        MU = k_size // 2 - shift
        MU = MU[None, None, :, None]

        # Create meshgrid for Gaussian
        [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]), indexing='ij')
        Z = np.stack([X, Y], 2)[:, :, :, None]

        # Calculate Gaussian for every pixel of the kernel
        ZZ = Z - MU
        ZZ[..., 0] *= -1
        ZZ_t = ZZ.transpose(0, 1, 3, 2)
        raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ))

        # Normalize the kernel and return
        kernel = raw_kernel / np.sum(raw_kernel)
        return kernel

    def get_gaussian2d_fit(self, kernel=None, location=None, index=None, n_samples=100000, return_parameters=False):
        assert(kernel is not None or location is not None or index is not None)

        def func_(x, k):
            mu = x[:2]
            sigma = x[2:].reshape((2, 2))
            k_size = np.array(k.shape)
            return (self.gaussian_filter_(SIGMA=sigma, shift=mu, k_size=k_size) - k).flatten()

        # If not given, get a kernel
        if location is not None:
            _, _, kernel = self.get_psf_by_location(location)
        if index is not None:
            _, _, kernel = self.get_psf_by_index(index)

        # ML naive fit
        density = kernel.reshape(-1, 3)
        density /= density.sum(axis=0)
        ker_size = self.filter_size
        I, J = np.meshgrid(-(np.arange(ker_size) - ker_size // 2), np.arange(ker_size) - ker_size // 2, indexing='ij')
        IJ = np.stack([I, J], axis=-1).reshape(-1, 2)
        IJ[..., 0] *= -1

        ij_r = np.random.choice(len(IJ), size=n_samples, replace=True, p=density[..., 0])
        X_r = IJ[ij_r, :]
        mean_r = np.mean(X_r, axis=0)
        var_r = (X_r - mean_r).T @ (X_r - mean_r) / n_samples
        mean_r = -mean_r
        # var_r = np.rot90(var_r, k=2)
        kernel_r = self.gaussian_filter_(var_r, shift=mean_r, k_size=np.array([ker_size, ker_size]))

        ij_g = np.random.choice(len(IJ), size=n_samples, replace=True, p=density[..., 1])
        X_g = IJ[ij_g, :]
        mean_g = np.mean(X_g, axis=0)
        var_g = (X_g - mean_g).T @ (X_g - mean_g) / n_samples
        mean_g = -mean_g
        # var_g = np.rot90(var_g, k=2)
        kernel_g = self.gaussian_filter_(var_g, shift=mean_g, k_size=np.array([ker_size, ker_size]))

        ij_b = np.random.choice(len(IJ), size=n_samples, replace=True, p=density[..., 2])
        X_b = IJ[ij_b, :]
        mean_b = np.mean(X_b, axis=0)
        var_b = (X_b - mean_b).T @ (X_b - mean_b) / n_samples
        mean_b = -mean_b
        # var_b = np.rot90(var_b, k=2)
        kernel_b = self.gaussian_filter_(var_b, shift=mean_b, k_size=np.array([ker_size, ker_size]))
        kernel_ml = np.stack([kernel_r, kernel_g, kernel_b], axis=-1)

        # LM robust fit
        func_r = lambda x: func_(x, kernel[..., 0])
        func_g = lambda x: func_(x, kernel[..., 1])
        func_b = lambda x: func_(x, kernel[..., 2])

        x0_r = np.concatenate([mean_r, var_r.flatten()])
        res_r = least_squares(func_r, x0=x0_r, method='lm')
        robust_mean_r = res_r.x[:2]
        robust_var_r = res_r.x[2:].reshape((2, 2))
        robust_var_r = 0.5 * (robust_var_r.T + robust_var_r)
        kernel_r = self.gaussian_filter_(robust_var_r, shift=robust_mean_r, k_size=np.array([ker_size, ker_size]))

        x0_g = np.concatenate([mean_g, var_g.flatten()])
        res_g = least_squares(func_g, x0=x0_g, method='lm')
        robust_mean_g = res_g.x[:2]
        robust_var_g = res_g.x[2:].reshape((2, 2))
        robust_var_g = 0.5 * (robust_var_g.T + robust_var_g)
        kernel_g = self.gaussian_filter_(robust_var_g, shift=robust_mean_g, k_size=np.array([ker_size, ker_size]))

        x0_b = np.concatenate([mean_b, var_b.flatten()])
        res_b = least_squares(func_b, x0=x0_b, method='lm')
        robust_mean_b = res_b.x[:2]
        robust_var_b = res_b.x[2:].reshape((2, 2))
        robust_var_b = 0.5 * (robust_var_b.T + robust_var_b)
        kernel_b = self.gaussian_filter_(robust_var_b, shift=robust_mean_b, k_size=np.array([ker_size, ker_size]))
        kernel_robust = np.stack([kernel_r, kernel_g, kernel_b], axis=-1)

        if return_parameters:
            theta = np.zeros(3)
            mu = np.zeros((3, 2))
            SIGMA_r = np.linalg.inv(robust_var_r)
            sigma_r, P_r = np.linalg.eig(SIGMA_r)
            mu[0] = np.array(center_of_mass(kernel_r))
            theta[0] = np.arcsin(P_r[1, 0])
            SIGMA_g = np.linalg.inv(robust_var_g)
            sigma_g, P_g = np.linalg.eig(SIGMA_g)
            mu[1] = np.array(center_of_mass(kernel_g))
            theta[1] = np.arcsin(P_g[1, 0])
            SIGMA_b = np.linalg.inv(robust_var_b)
            sigma_b, P_b = np.linalg.eig(SIGMA_b)
            mu[2] = np.array(center_of_mass(kernel_b))
            theta[2] = np.arcsin(P_b[1, 0])
            mu -= mu[1, :]  # make sure green is at (0,0)
            sigma = np.stack([sigma_r[::-1], sigma_g[::-1], sigma_b[::-1]], axis=0)
            parameters = {'theta': theta, 'sigma': sigma, 'mu': mu}
            return kernel_ml, kernel_robust, parameters
        else:
            return kernel_ml, kernel_robust


if __name__ == "__main__":
    psf_path = './psfs/Canon_EF24mm_f_1.4L_USM_ap_1.4.mat'
    psf = PSF(psf_path)
    print(psf)

    import matplotlib.pyplot as plt
    # plt.figure()
    # loc, exp, k = psf.get_psf_by_location((300,300))
    # loc, exp, k = psf.get_psf_by_location((400, 2000))
    # loc, exp, k = psf.get_psf_by_location((1300, 2300))
    loc, exp, k = psf.get_psf_by_location((3300, 2300))
    # loc, exp, k = psf.get_psf_by_location((4300, 100))
    # plt.imshow(k / k.max())
    # plt.title('(%d, %d), EV=%d' % (*loc, exp))
    # plt.show()

    k_ml, k_robust = psf.get_gaussian2d_fit(k)
    plt.imshow(k / k.max())
    plt.subplot(1, 3, 1)
    plt.imshow(k / k.max())
    plt.title('(%d, %d), EV=%d' % (*loc, exp))
    plt.subplot(1, 3, 2)
    plt.imshow(k_ml / k_ml.max())
    plt.title('MLE')
    plt.subplot(1, 3, 3)
    plt.imshow(k_robust / k_robust.max())
    plt.title('Robust')
    plt.show()
