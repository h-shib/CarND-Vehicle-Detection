import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
from collections import deque
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from time import time
from moviepy.editor import VideoFileClip

class VehicleDetector:
    def __init__(self):
        self.clf = None
        self.scaler = None
        self.frames = deque([])

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
        if vis:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis, feature_vector=feature_vec)
            return features

    def get_scaler(self, features):
        # return scaler and save it
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
        window_list = []
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
        hot_windows = []

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch_base = self.convert_img_color(img_tosearch, cspace=cspace)

        scales = [1, 1.2, 1.5]
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
            cells_per_step = 2 # Instead of overlap, define how many cells to step
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
                        hot_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
        return draw_img, hot_windows

    def apply_head_threshold(self, heatmap, bboxes, threshold):
        for box in bboxes:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        heatmap[heatmap<=threshold] = 0
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

    def process_image(self, image):
        draw_img = np.copy(image)
        image = image.astype(np.float32)/255

        windows = []
        for i in range(len(xy_windows)):
            x_start_stop = x_start_stops[i]
            y_start_stop = y_start_stops[i]
            xy_window = xy_windows[i]
            windows.extend(vd.slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap))

        hot_windows = vd.search_windows(image, windows, clf, scaler, cspace=cspace, spatial_size=spatial_size, hist_nbins=hist_nbins, hist_bins_range=hist_bins_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, hog_channel=hog_channel)

        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = vd.apply_head_threshold(heat, hot_windows, 1)

        self.frames.append(heat)
        heat_result = heat
        if len(self.frames) > 1:
            for i in range(len(self.frames)-1):
                heat_result = np.add(heat_result, self.frames[i])
            if len(self.frames) > 7:
                self.frames.popleft()

        labels = label(heat_result)
        draw_img = vd.draw_labeled_bboxes(draw_img, labels)
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
    update_models = True
    cspace = 'YCrCb'         # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9               # HOG orientations
    pix_per_cell = 8         # HOG pixels per cell
    cell_per_block = 2       # HOG cells per block
    hog_channel = 'ALL'      # Can be 0, 1, 2, or "ALL"
    spatial_size = (64, 64)  # Spatial binning dimensions
    hist_nbins = 32          # Number of histogram bins
    hist_bins_range = (0, 1) # Bins range of histogram
    vis = False              # visualization on or off
    feature_vec = True       # return features 
    spatial_feat = True      # Spatial features on or off
    hist_feat = True         # Histogram features on or off
    hog_feat = True          # HOG features on or off

    x_start_stops = [[350, 1250], # window search area x
                     [200, 1250],
                     [200, 1250],
                     [50, 1250],
                     [50, 1250]]
    y_start_stops = [[400, 496], # window search area y
                     [400, 544],
                     [400, 592],
                     [400, 640],
                     [400, 650]]
    xy_windows    = [(64, 64), # window sizes
                     (96, 96),
                     (128, 128),
                     (160, 160),
                     (200, 200)]
    xy_overlap    = (0.75, 0.75) # overlap area


    # initialize detector and train classifier and scaler
    vd = VehicleDetector() # TODO: set initial parameters
    clf, scaler = vd.get_trained_models(car_fnames, non_car_fnames, update_models=update_models, cspace=cspace, spatial_size=spatial_size, hist_nbins=hist_nbins, hist_bins_range=hist_bins_range, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis, feature_vec=feature_vec, hog_channel=hog_channel)

    # process each image on video
    output = 'project_video_result.mp4'
    clip_input = VideoFileClip("project_video.mp4")
    clip = clip_input.fl_image(lambda x: vd.process_image(x))
    clip.write_videofile(output, audio=False)
