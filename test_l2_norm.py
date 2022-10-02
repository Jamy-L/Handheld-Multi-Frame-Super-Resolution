import numpy as np
from scipy import signal
from skimage import img_as_float32, data

# parameters
dtype = np.float32

x = 50
y = 50
n = 16
r = 10
m = n + 2* r
t = m - n + 1

hn = n//2
hm = m//2
ht = t//2

# init
img = img_as_float32(data.astronaut())
img = img.mean(axis=-1)

img_shifted = np.roll(img, (3,3))

# the for loop way



# the naive way
T = img[x-hn:x+hn, y-hn:y+hn]  # (n,n)
I = img_shifted[x-hm:x+hm, y-hm:y+hm]  # (m,m)

T_norm = np.linalg.norm(T, ord=2)  # (1,1)

box = np.ones((n, n))  # (n,n) -> same size as T
I_box_ = signal.convolve2d(I*I, box, mode='valid')  # (m-n+1,m-n+1) or (t,t)

IT_cross_ = signal.correlate2d(I, T, mode='valid')  # (m-n+1,m-n+1)

res_ = -2 * IT_cross_ + I_box_ + T_norm


# the FFT way
# T is the same as before

box_fft = np.zeros((m,m))
box_fft[:n,:n] = 1
box_fft = np.fft.fft2(box_fft)
I_box = np.fft.ifft2(np.fft.fft2(I*I) * box_fft).real
I_box = I_box[n-1:n-1+t, n-1:n-1+t]
# print(I_box)
# print(I_box_)
# print(I_box.shape, I_box_.shape)
print('diff box:', np.linalg.norm(I_box - I_box_))

fI = np.fft.fft2(I)
fT = np.fft.fft2(T, s=I.shape)
IT_cross = np.fft.ifft2(fT.conjugate() * fI).real
IT_cross = IT_cross[:t, :t]

print('diff cross:', np.linalg.norm(IT_cross - IT_cross_))
# print('mine', IT_cross)
# print('expected', IT_cross_)
