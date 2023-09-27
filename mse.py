import numpy as np
import cv2

imagen_original_1 = cv2.imread('editadisima.jpg')
imagen_original_2 = cv2.imread('generadisima.jpg')

mse = np.mean((imagen_original_1 - imagen_original_2)**2)
print(mse)
