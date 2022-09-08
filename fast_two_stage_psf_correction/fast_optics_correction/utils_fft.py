import torch
import torch.fft
import numpy as np


def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxHx2
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


def c2c(x):
    return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)


def crdiv(x, y):
    # complex/real division
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a/y, b/y], -1)


def csum(x, y):
    # complex + real
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    # modulus of a complex number
    return torch.pow(x[..., 0]**2+x[..., 1]**2, 0.5)


def cabs2(x):
    return x[..., 0]**2+x[..., 1]**2


def cmul(t1, t2):
    '''complex multiplication
    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2
    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def creal(t):
    return t[..., 0]


def cimag(t):
    return t[..., 1]


def cconj(t, inplace=False):
    '''complex's conjugation
    Args:
        t: NxCxHxWx2
    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def rfft(t):
    # Real-to-complex Discrete Fourier Transform
    return torch.fft.rfft(t, 2, onesided=False)


def irfft(t):
    # Complex-to-real Inverse Discrete Fourier Transform
    return torch.fft.irfft(t, 2, onesided=False)


def fft(t):
    # Complex-to-complex Discrete Fourier Transform
    return torch.fft.fft(t, 2)


def ifft(t):
    # Complex-to-complex Inverse Discrete Fourier Transform
    return torch.fft.ifft(t, 2)


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        # otf: NxCxHxWx2
        otf: NxCxHxW
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    # otf = torch.rfft(otf, 2, onesided=False)
    otf = torch.fft.fft2(otf, dim=(-2, -1))
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, *xCxW/sfxH/sf
    z: tensor image, *xCxWxH
    '''
    st = 0
    z = torch.zeros((*x.shape[0:-3], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def upsample_raw(x):
    ''' Upsampling an RGGB-formated RAW image (equivalent to ``atrous'' image).
    x: tensor RAW image, *x4xH/2xW/2
    y: tensor RAW image, *x4xHxW
    '''
    z = torch.zeros((*x.shape[0:-3], x.shape[1], x.shape[2]*2, x.shape[3]*2)).type_as(x)
    z[..., 0:1, 0::2, 0::2].copy_(x[...,0:1,:,:])
    z[..., 1:2, 0::2, 1::2].copy_(x[...,1:2,:,:])
    z[..., 2:3, 1::2, 0::2].copy_(x[...,2:3,:,:])
    z[..., 3:4, 1::2, 1::2].copy_(x[...,3:4,:,:])
    return z


def downsample(x, sf=3):
    '''s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]

