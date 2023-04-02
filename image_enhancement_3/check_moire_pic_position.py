import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('./magnitude_spectrum_back.png', 0)

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

def onclick(event):
    x = int(event.xdata)
    y = int(event.ydata)
    print('Clicked on pixel (', x, ', ', y, ') with magnitude ', magnitude_spectrum[y, x])
    
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

plt.show()






