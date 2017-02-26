import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from time import time

class VehicleDetector:
    def __init_(self):
        self.detector_model = None

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

    def normalize_features(self, features):
        X_scaler = StandardScaler().fit(features)
        scaled_features = X_scaler.transform(features)
        return scaled_features

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
        features = np.vstack(features).astype(np.float64)
        return features

    def train_model(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
        svc = LinearSVC()
        t1 = time()
        svc.fit(X_train, y_train)
        t2 = time()
        print(round(t2-t1, 2), "seconds to train model.")
        model_fname = 'models/vehicle_detect_model.p'
        pickle.dump(svc, open(model_fname, 'wb'), protocol=4)
        print("Model saved as: ", model_fname)
        print("Test Accuracy: ", svc.score(X_test, y_test))
        return svc

    def get_trained_model(self, car_fnames, non_car_fnames, update=False):
        # use pickled model if exists
        if not update and os.path.exists('models/vehicle_detect_model.p'):
            self.model = pickle.load(open('models/vehicle_detect_model.p', 'rb'))
            print('used exist model.')
            return self.model

        # train new model
        car_features = self.extract_features(car_fnames)
        non_car_features = self.extract_features(non_car_fnames)
        features = np.vstack((car_features, non_car_features)).astype(np.float64) 
        scaled_features = self.normalize_features(features)
        labels = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))
        self.model = self.train_model(scaled_features, labels)
        print('trained and set new model.')
        return self.model

    def process_frame(self, img):
        pass



if __name__ == '__main__':

    input_dir = 'test_images'
    """
    for fname in os.listdir(input_dir):
        #image = mpimg.imread(os.path.join(input_dir, fname))
        fname = os.path.join(input_dir, fname)
        vd = VehicleDetector()
        #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fnames = [fname]
        res = vd.extract_features(fnames)
        print(res[0].nonzero())
        #plt.imshow(image)
        #plt.show()
    """

    car_dir = 'vehicles'
    non_car_dir = 'non-vehicles'

    # extract car file names
    car_fnames = []
    for directory in os.listdir(car_dir):
        for fname in os.listdir(os.path.join(car_dir, directory)):
            car_fnames.append(os.path.join(car_dir, directory, fname))

    # extract non car file names
    non_car_fnames = []
    for directory in os.listdir(non_car_dir):
        for fname in os.listdir(os.path.join(non_car_dir, directory)):
            non_car_fnames.append(os.path.join(non_car_dir, directory, fname))


    vd = VehicleDetector()
    vd.get_trained_model(car_fnames, non_car_fnames, True)
    #res = vd.extract_features(car_fnames, normalize=True)













