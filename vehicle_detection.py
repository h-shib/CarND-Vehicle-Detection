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
        self.clf = None
        self.scaler = None

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

    def get_scaler(self, features):
        X_scaler = StandardScaler().fit(features)
        scaler_fname = 'models/vehicle_detect_scaler.p'
        pickle.dump(X_scaler, open(scaler_fname, 'wb'))
        print("Scaler saved as: ", scaler_fname)
        return X_scaler

    def extract_features(self, fnames, cspace='RGB', spatial_size=(32, 32), hist_nbins=32, hist_bins_range=(0, 256), orient=9, pix_per_cell=8,
                    cell_per_block=2, vis=False, feature_vec=True, hog_channel='ALL'):
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
            if hog_channel == 'ALL':
                hog_features = []
                for ch in range(feature_img.shape[2]):
                    hog_features.append(self.get_hog_features(feature_img[:, :, ch],orient=orient, pix_per_cell=pix_per_cell,
                    cell_per_block=cell_per_block, vis=vis, feature_vec=feature_vec))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = self.get_hog_features(feature_img,orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis, feature_vec=feature_vec)
            features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        features = np.vstack(features).astype(np.float64)
        return features

    def extract_single_image_features(self, img, cspace='RGB', spatial_size=(32, 32), hist_nbins=32, orient=9, pix_per_cell=8, cell_per_block=2, spatial_feat=True, hist_feat=True, hog_feat=True, hog_channel='ALL'):
        img_features = []
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_img = np.copy(img)
        if spatial_feat:
            spatial_features = cv2.resize(feature_img, spatial_size).ravel()
            img_features.append(spatial_features)
        if hist_feat:
            hist_features = self.color_hist(feature_img, nbins=hist_nbins)
            img_features.append(hist_features)
        if hog_feat:
            if hog_channel == 'ALL':
                hog_features = []
                for ch in range(feature_img.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_imag[:,:,ch], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            else:
                hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            img_features.append(hog_features)
        return np.concatenate(img_features)

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def search_windows(self, img, windows, clf, scaler, cspace='RGB', spatial_size=(32, 32), hist_nbins=32, hist_bins_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, spatial_feat=True, hist_feat=True, hog_feat=True, hog_channel=0):
        on_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = self.extract_single_image_features(test_img, cspace=cspace, spatial_size=spatial_size, hist_nbins=hist_nbins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            prediction = clf.predict(test_features)
            if prediction == 1:
                on_windows.append(window)
        return on_windows

    def train_model(self, features, labels):
        # split features and labels into training and test sets.
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

        # train the model
        svc = LinearSVC()
        print("start training model.")
        t1 = time()
        svc.fit(X_train, y_train)
        t2 = time()
        print(round(t2-t1, 2), "seconds to train model.")
        print("Test Accuracy: ", svc.score(X_test, y_test))

        # save the model as pickle file.
        model_fname = 'models/vehicle_detect_model.p'
        pickle.dump(svc, open(model_fname, 'wb'), protocol=4)
        print("Model saved as: ", model_fname)
        return svc

    def get_trained_models(self, car_fnames, non_car_fnames, update=False):
        # use pickled classifier and scaler if exists
        if not update and os.path.exists('models/vehicle_detect_clf.p') and os.path.exists('models/vehicle_detect_scaler.p'):
            clf = pickle.load(open('models/vehicle_detect_clf.p', 'rb'))
            scaler = pickle.load(open('models/vehicle_detect_scaler.p', 'rb'))
            self.clf = clf
            self.scaler = scaler
            print('used exist models.')
            return clf, scaler

        # train new model
        car_features = self.extract_features(car_fnames)
        non_car_features = self.extract_features(non_car_fnames)
        features = np.vstack((car_features, non_car_features)).astype(np.float64)
        scaler = self.get_scaler(features)
        scaled_features = scaler.transform(features)
        labels = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))
        clf = self.train_model(scaled_features, labels)
        self.clf = clf
        self.scaler = scaler
        print('trained and set new model.')
        return clf, scaler

    def process_frame(self, img):
        pass


if __name__ == '__main__':

    input_dir = 'test_images'
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
    clf, scaler = vd.get_trained_models(car_fnames, non_car_fnames)

    image = mpimg.imread('test_images/test1.jpg')
    windows = vd.slide_window(image)
    scaler = vd.get_scaler()
    hot_windows = vd.search_windows(image, windows, vd.clf, vd.scaler)
    print(hot_windows)




