import cv2
import numpy as np

depth = cv2.imread('depth.png', cv2.IMREAD_UNCHANGED)

# normalize
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = np.uint8(depth_norm)


depth_eq = cv2.equalizeHist(depth_norm)
depth_color = cv2.applyColorMap(depth_eq, cv2.COLORMAP_JET)

cv2.imwrite('depth_contrast.png', depth_eq)
cv2.imwrite('depth_jet.png', depth_color)


