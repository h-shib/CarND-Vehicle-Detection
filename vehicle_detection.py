import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog


class VehicleDetector:
    def __init_(self):
        pass

    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        """
        Compute the histogram of the color channels
        and return concatenated single feature vector
        """
        c1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        c2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        c3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        return np.concatenate((c1_hist[0], c2_hist[0], c3_hist[0]))

    def get_hog_features(self, img, orient=9, pix_per_cell=8,
                    cell_per_block=2, vis=False, feature_vec=True):
        """
        return HOG features and visualization
        if vis == True: return features, hog_image
        else:           return features
        """
        return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis, feature_vector=feature_vec)

    def extract_features(self, fnames, cspace='RGB', spatial_size=(32, 32), hist_nbins=32, hist_bins_range=(0, 256)):
        features = []
        for fname in fnames:
            img = mpimg.imread(fname)
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            else:
                feature_img = np.copy(img)
            spatial_features = cv2.resize(feature_img, spatial_size).ravel()
            hist_features = self.color_hist(feature_img, nbins=hist_nbins, bins_range=hist_bins_range)
            features.append(np.concatenate((spatial_features, hist_features)))
        return features



if __name__ == '__main__':

    input_dir = 'test_images'
    for fname in os.listdir(input_dir):
        #image = mpimg.imread(os.path.join(input_dir, fname))
        fname = os.path.join(input_dir, fname)
        vd = VehicleDetector()
        #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fnames = [fname]
        res = vd.extract_features(fnames)
        print(res)
        #plt.imshow(image)
        #plt.show()