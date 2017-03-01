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

    def convert_img_color(self, img, cspace='HLS'):
        if cspace == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if cspace == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if cspace == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img[:,:,0] = img[:,:,0]/360
            return img
        if cspace == 'HLS':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            img[:,:,0] = img[:,:,0]/360
            return img
        return np.copy(img)

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

    def extract_features(self, fnames, cspace='RGB', spatial_size=(32, 32), hist_nbins=32, hist_bins_range=(0, 256), orient=9, pix_per_cell=8,cell_per_block=2, vis=False, feature_vec=True, hog_channel='ALL'):
        features = []
        for fname in fnames:
            img = mpimg.imread(fname)
            feature_img = self.convert_img_color(img, cspace)
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

    def extract_single_image_features(self, img, cspace='RGB', spatial_size=(32, 32), hist_nbins=32, hist_bins_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, spatial_feat=True, hist_feat=True, hog_feat=True, hog_channel='ALL'):
        img_features = []
        feature_img = self.convert_img_color(img, cspace)
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
                    hog_features.extend(self.get_hog_features(feature_img[:,:,ch], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            else:
                hog_features = self.get_hog_features(feature_img[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
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

    def search_windows(self, img, windows, clf, scaler, cspace='RGB', spatial_size=(32, 32), hist_nbins=32, hist_bins_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, spatial_feat=True, hist_feat=True, hog_feat=True, hog_channel='ALL'):
        on_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

            features = self.extract_single_image_features(test_img, cspace=cspace, spatial_size=spatial_size, hist_nbins=hist_nbins, hist_bins_range=hist_bins_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
            scaled_features = scaler.transform(np.array(features).reshape(1, -1))
            prediction = clf.predict(scaled_features)
            if prediction == 1:
                on_windows.append(window)
        return on_windows

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

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
        model_fname = 'models/vehicle_detect_clf.p'
        pickle.dump(svc, open(model_fname, 'wb'), protocol=4)
        print("Model saved as: ", model_fname)
        return svc

    def get_trained_models(self, car_fnames, non_car_fnames, update_models=False, cspace='RGB', spatial_size=(32, 32), hist_nbins=32, hist_bins_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True, hog_channel='ALL'):
        # use pickled classifier and scaler if exists
        if not update_models and (os.path.exists('models/vehicle_detect_clf.p') and os.path.exists('models/vehicle_detect_scaler.p')):
            clf = pickle.load(open('models/vehicle_detect_clf.p', 'rb'))
            scaler = pickle.load(open('models/vehicle_detect_scaler.p', 'rb'))
            self.clf = clf
            self.scaler = scaler
            print('used exist models.')
            return clf, scaler

        # train new model
        print('start updating the models.')
        car_features = self.extract_features(car_fnames, cspace=cspace, spatial_size=spatial_size, hist_nbins=hist_nbins, hist_bins_range=hist_bins_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis, feature_vec=feature_vec, hog_channel=hog_channel)
        non_car_features = self.extract_features(non_car_fnames, cspace=cspace, spatial_size=spatial_size, hist_nbins=hist_nbins, hist_bins_range=hist_bins_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis, feature_vec=feature_vec, hog_channel=hog_channel)
        features = np.vstack((car_features, non_car_features)).astype(np.float64)
        scaler = self.get_scaler(features)
        scaled_features = scaler.transform(features)
        labels = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))
        clf = self.train_model(scaled_features, labels)
        self.clf = clf
        self.scaler = scaler
        print('trained and set new model.')
        return clf, scaler

    def find_cars(self, img, ystart, ystop, scale, clf, scaler, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
        draw_img = np.copy(img)

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch_base = self.convert_img_color(img_tosearch, cspace=cspace)

        scales = [1, 1.5]
        for scale in scales:
            if scale != 1:
                imshape = ctrans_tosearch_base.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch_base, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            else:
                ctrans_tosearch = ctrans_tosearch_base

            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // pix_per_cell)-1
            nyblocks = (ch1.shape[0] // pix_per_cell)-1 
            nfeat_per_block = orient*cell_per_block**2
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // pix_per_cell)-1 
            cells_per_step = 2  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step
            
            # Compute individual channel HOG features for the entire image
            hog1 = self.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = self.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = self.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
            
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    xleft = xpos*pix_per_cell
                    ytop = ypos*pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                    # Get color features
                    spatial_features = cv2.resize(subimg, spatial_size).ravel()
                    hist_features = self.color_hist(subimg, nbins=hist_bins)

                    # Scale features and make a prediction
                    test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

                    #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                    test_prediction = clf.predict(test_features)
                    
                    if test_prediction == 1:
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
        return draw_img



if __name__ == '__main__':

    input_dir = 'test_images'
    car_dir = 'vehicles'
    non_car_dir = 'non-vehicles'

    # extract car file names
    print('start reading file names.')
    car_fnames = []
    for directory in os.listdir(car_dir):
        for fname in os.listdir(os.path.join(car_dir, directory)):
            car_fnames.append(os.path.join(car_dir, directory, fname))
    print('finish reading car file names.')

    # extract non car file names
    non_car_fnames = []
    for directory in os.listdir(non_car_dir):
        for fname in os.listdir(os.path.join(non_car_dir, directory)):
            non_car_fnames.append(os.path.join(non_car_dir, directory, fname))
    print('finish reading non-car file names.')

    # parameter settings
    update_models = False
    cspace = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_nbins = 16    # Number of histogram bins
    hist_bins_range = (0, 255)
    vis = False
    feature_vec = True
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    #x_start_stops = [[400, 1000], [300, 1100], [200, 1200], [100, None]]
    x_start_stops = [[100, None], [100, None], [100, None], [100, None]]
    y_start_stop = (400, 650) # Min and max in y to search in slide_window()
    xy_overlap = (0.5, 0.5)
    xy_windows = [(64, 64), (96, 96), (128, 128), (160, 160)]


    vd = VehicleDetector()
    clf, scaler = vd.get_trained_models(car_fnames, non_car_fnames, update_models=update_models, cspace=cspace, spatial_size=spatial_size, hist_nbins=hist_nbins, hist_bins_range=hist_bins_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis, feature_vec=feature_vec, hog_channel=hog_channel)

    image = mpimg.imread('test_images/test1.jpg')
    image = image.astype(np.float32)/255 # convert to 0 to 1
    print(image.shape)

    windows = []
    for i in range(len(xy_windows)):
        x_start_stop = x_start_stops[i]
        xy_window = xy_windows[i]
        windows.extend(vd.slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap))

    hot_windows = vd.search_windows(image, windows, clf, scaler, cspace=cspace, spatial_size=spatial_size, hist_nbins=hist_nbins, hist_bins_range=hist_bins_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, hog_channel=hog_channel)
    print(len(hot_windows))

    ystart = 400
    ystop = 650
    scale = 1.5

    window_img = vd.draw_boxes(image, hot_windows, thick=6)
    window_img = vd.find_cars(image, ystart, ystop, scale, clf, scaler, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_nbins)
    plt.imshow(window_img)
    plt.show()




