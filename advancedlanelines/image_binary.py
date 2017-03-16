import cv2
import numpy as np


class ImageBinarizer:
    def __init__(self, thresh_min, thresh_max, s_thresh_min, s_thresh_max):
        self.thresh_max = thresh_max
        self.thresh_min = thresh_min
        self.s_thresh_min = s_thresh_min
        self.s_thresh_max = s_thresh_max

    def binarize(self, img):
        """
        This method binarizes with combined s channel and gradient thresholds
        """
        s_binary = self.s_channel_binary(img)
        sxbinary = self.gradient_threshold_binary(img)
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        # scale up so that the binary image draws properly else 0 and 1 are represented by black.
        combined_binary = 255 * np.dstack(
            (combined_binary, combined_binary, combined_binary)
        ).astype('uint8')
        return combined_binary

    def s_channel_binary(self, img):
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        # compute the binary image based on threshold values
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh_min) & (s_channel <= self.s_thresh_max)] = 1
        return s_binary

    def gradient_threshold_binary(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.thresh_min) & (scaled_sobel <= self.thresh_max)] = 1
        return sxbinary

    def binarize_given_path(self, img_path):
        return self.binarize(img=cv2.imread(img_path))

    def binarize_given_path_and_save(self, img_path, out_path):
        binary = self.binarize(img=cv2.imread(img_path))
        cv2.imwrite(out_path, binary)
        return binary