
import numpy as np
import cv2
import sys
import pickle

from sklearn.preprocessing import StandardScaler
import time
from sklearn.svm import LinearSVC

from color_feature import *

import time
from scipy.ndimage.measurements import label

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    The function draws bounding boxes from the list on the image
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    count = 0
    print(imcopy.dtype, imcopy.shape)

    #cv2.imshow("1", imcopy)
    # Iterate through the bounding boxes
    for bbox in bboxes:

        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color , thick)
        count += 1

    print(count)
    # Return the image copy with boxes drawn
    return imcopy


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    The function creates sliding window given
    start and stop co-ordinates of rows and coloumns of the image
    """
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

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    #print(window_list)
    # Return the list of windows
    return window_list


def detect_object(img, classifier, normalizer,features_cfg,
                  scale = (1, 1), x_start_stop=[None, None], y_start_stop=[None, None],
                  xy_window=(64, 64), xy_overlap=(0.5, 0.5), fast_hog_extract = 1):

    """
    This function runs the svm classification on the the ROI
    and outputs detected windows
    """

    assert (len(img.shape) == 3)

    x_scale = scale[0]
    y_scale = scale[1]

    if ((x_scale  !=  1) or (y_scale != 1)):

        scaled_x = int(img.shape[1] * x_scale)
        scaled_y = int(img.shape[0] * y_scale)
        rescaled_img = cv2.resize(img, (scaled_x, scaled_y))

        if y_start_stop[0] != None:
            y_start_stop[0] = int (y_start_stop[0] * y_scale)

        if y_start_stop[1] != None:
            y_start_stop[1] = int (y_start_stop[1] * y_scale)

        if x_start_stop[0] != None:
            x_start_stop[0] = int (x_start_stop[0] * x_scale)

        if x_start_stop[1] != None:
            x_start_stop[1] = int (x_start_stop[1] * x_scale)

    else:
        rescaled_img = img



    print("!!", rescaled_img.shape, x_start_stop, y_start_stop, xy_window, xy_overlap)
    windows = slide_window(rescaled_img, x_start_stop, y_start_stop,
                    xy_window, xy_overlap)

    detect_wins = []
    scaled_detect_wins = []

    if  fast_hog_extract == 1:
        img_x_start = x_start_stop[0]
        img_y_start = y_start_stop[0]

        img_x_end = x_start_stop[1]
        img_y_end = y_start_stop[1]

        if (img_x_start == None):
            img_x_start = 0
        if (img_y_start == None):
            img_y_start = 0
        if (img_x_end == None):
            img_x_end = img.shape[1]
        if (img_y_end == None):
            img_y_end = img.shape[0]

        hog = HoG_feature()
        hog.feature_vector = False
        hog.channel = "ALL"

        #print("---", img_y_start, img_y_end, img_x_start, img_x_end)
        hog_rescaled_img = rescaled_img[img_y_start:img_y_end, img_x_start:img_x_end]
        #hog_rescaled_img = rescaled_img[img_y_start:img_y_end, img_x_start:img_x_end]
        feature_image = convert_color_space(hog_rescaled_img, features_cfg.hog_cs)
        ch0_hog_features, ch1_hog_features, ch2_hog_features = extract_HoG_fvector(feature_image, hog)
        #print(ch0_hog_features.shape)
        #print(ch1_hog_features.shape)
        #print(ch2_hog_features.shape)


    #print(output_fvec.shape)
    total_extract_time = 0
    for win in windows:
        start_cord = win[0]
        end_cord = win[1]

        start_x = start_cord[0]
        start_y = start_cord[1]

        end_x = end_cord[0]
        end_y = end_cord[1]
        #print(win, start_cord, end_cord)

        #extract the image patch
        img_patch = rescaled_img[start_y:end_y, start_x:end_x, :]

        #print(img_patch.shape)
        if(img_patch.shape != (64, 64, 3)):
            print("resize image patch")
            img_patch = cv2.resize(img_patch, (64, 64))

        #print(img_patch.shape)
        #cv2.imshow("img_patch", img_patch)
        #get features vector for this patch
        #print(features_cfg)
        if  fast_hog_extract != 1:
            t_start = time.time()
            output_fvec = extract_img_features(img_patch, features_cfg)
            t_end = time.time()
            total_extract_time += (t_end - t_start)
        else:

            t_start = time.time()
            hog_features = []
            features_cfg.hog_features = False
            fvec_spatial_hist = extract_img_features(img_patch, features_cfg)
            #get hog features
            #def extract_hog_features(hog_img_ch, start_pt = (0, 0), end_pt = (64, 64), pix_per_cell=(8, 8), cells_per_blk = (2, 2)):
            start_x_hog = start_cord[0]
            start_y_hog = start_cord[1]

            end_x_hog = end_cord[0]
            end_y_hog = end_cord[1]

            start_x_hog = start_x_hog - img_x_start
            start_y_hog = start_y_hog - img_y_start
            end_x_hog = end_x_hog - img_x_start
            end_y_hog = end_y_hog - img_y_start

            ch0_hog_fv = extract_hog_features(ch0_hog_features, (start_x_hog, start_y_hog), (end_x_hog, end_y_hog))
            hog_features.append(ch0_hog_fv)
            ch1_hog_fv = extract_hog_features(ch1_hog_features, (start_x_hog, start_y_hog), (end_x_hog, end_y_hog))
            hog_features.append(ch1_hog_fv)
            ch2_hog_fv = extract_hog_features(ch2_hog_features, (start_x_hog, start_y_hog), (end_x_hog, end_y_hog))
            hog_features.append(ch2_hog_fv)

            output_fvec = np.concatenate((fvec_spatial_hist, ch0_hog_fv, ch1_hog_fv, ch2_hog_fv))
            t_end = time.time()
            total_extract_time += (t_end - t_start)
            #print(output_fvec.shape, fvec_spatial_hist.shape)

        output_fvec_array = np.array(output_fvec).reshape(1,-1)
        #print(output_fvec_array[0])

        # Apply the scaler to X
        scaled_output_fvec = normalizer.transform(output_fvec_array)

        #print(output_fvec.shape, scaled_output_fvec.shape)

        #print(output_fvec[0], scaled_output_fvec[0])

        pred = classifier.predict(scaled_output_fvec)
        if pred == 1:
            start_x = int(start_x / x_scale)
            start_y = int(start_y / y_scale)

            end_x = int(end_x / x_scale)
            end_y = int(end_y / y_scale)

            scaled_win = ((start_x, start_y), (end_x, end_y))
            detect_wins.append(win)
            scaled_detect_wins.append(scaled_win)
        #break;


        #print("pred is for window: ", win, pred)

        #break;

    #window_img = draw_boxes(rescaled_img, detect_wins, color=(0, 255, 0), thick=4)
    #cv2.imshow("rescaled_img", window_img)

    print("total feature extract time {0:}".format(total_extract_time))
    return scaled_detect_wins, detect_wins, rescaled_img

def test_slide_window(img, multi_scale_cfg):

    wins = []
    for level in multi_scale_cfg[:4]:
        print(level)

        level_id = level[0]
        scale = level[1]
        x_start_stop = level[2]
        y_start_stop = level[3]
        rescaled_xy_window = int(level[4][0]/scale[0]), int(level[4][1]/scale[1])
        print(rescaled_xy_window)
        xy_overlap = level[5]
        xy_overlap = (0, 0)

        windows = slide_window(img, x_start_stop, y_start_stop,
                    rescaled_xy_window, xy_overlap)
        wins.extend(windows)
        #window_img = draw_boxes(img, windows, color=(0, 255, 0), thick=4)
        #cv2.imshow("window_scale_img" + level_id, window_img)
        #cv2.imwrite("./window_scale_img_ax_ay" + level_id + ".png", window_img)

    window_img = draw_boxes(img, wins, color=(0, 255, 0), thick=4)
    cv2.imshow("window_scale_img" + level_id, window_img)
    cv2.imwrite("./window_multiscale_img_0x_0y" + level_id + ".png", window_img)



def detect_object_multiscale(img, svc_classify, norm, features, multi_scale_cfg, fast_hog_extract = 1, draw_win_flag=0):

    """
    The function performs mutiscale object detection on the image frame
    """
    windows = []
    total_time = 0
    for level in multi_scale_cfg[:3]:
        print(level)

        level_id = level[0]
        scale = level[1]
        x_start_stop = level[2]
        y_start_stop = level[3]
        xy_window = level[4]
        xy_overlap = level[5]

        #print(level_id, scale, x_start_stop, y_start_stop, xy_window, xy_overlap)

        #print(y_start_stop)
        #y_start_stop[0] = int (img.shape[0] / 3)
        t_start =time.time()
        rescaled_detect_wins,  ori_detect_wins, rescaled_img = detect_object(img, svc_classify, norm, features, scale,
                                                                              x_start_stop, y_start_stop,
                                                                             xy_window, xy_overlap, fast_hog_extract)

        t_end = time.time()

        print("Time_level:{0:}".format(t_end-t_start))

        # writing and displaying multi scale detection
        #window_img = draw_boxes(img, rescaled_detect_wins, color=(0, 255, 0), thick=4)
        #cv2.imshow("window_img" + level_id, window_img)
        #cv2.imwrite("./window_img" + level_id + ".png", window_img)

        #rescaled_window_img = draw_boxes(rescaled_img, ori_detect_wins, color=(0, 255, 0), thick=4)
        #cv2.imshow("window_img_rescaled" + level_id, rescaled_window_img)

        #windows.append(rescaled_detect_wins)
        windows.extend(rescaled_detect_wins)
        total_time += (t_end-t_start)


    print("total time:{0:}".format(total_time))
    window_img = img
    #for win_scale in windows:
    win_scale = windows
    #print(window_img.shape)
    if draw_win_flag == 1:
        window_img = draw_boxes(window_img, win_scale, color=(0, 255, 0), thick=4)
        #cv2.imshow("window_img_multi", window_img)

    return windows, window_img

def get_multi_scale_config(img):

        """
        configuration for multiscale slinding windows
        """
        multiscale_cfg = []

        scale=(1, 1)
        x_start_stop=[None, None]
        y_start_stop=[None, None]
        xy_window=(64, 64)
        xy_overlap=(1 - 2/8, 1 - 2/8)
        y_start_stop[0] = int (img.shape[0] / 2)
        y_start_stop[1] = y_start_stop[0] + int(1 * (img.shape[0] - y_start_stop[0]) / 3)
        y_start_stop[1] = min(y_start_stop[1], img.shape[0])

        multiscale_cfg.append(["scale_0", scale,
                               x_start_stop,
                               y_start_stop,
                               xy_window,
                               xy_overlap]
                             )

        scale = (64/ (64*(1.5**1) ), 64/( 64*(1.5**1) ))
        x_start_stop=[None, None]
        y_start_stop=[None, None]
        xy_window=(64, 64)
        xy_overlap=(1 - 1/8, 1 - 1/8)
        y_start_stop[0] = int (img.shape[0] / 3 + 128)
        y_start_stop[0] = int (img.shape[0] / 2 + 32)

        y_start_stop[1] = y_start_stop[0] + int(1 * (img.shape[0] - y_start_stop[0]) / 3)
        y_start_stop[1] = min(y_start_stop[1], img.shape[0])

        multiscale_cfg.append(["scale_1", scale,
                                               x_start_stop,
                                               y_start_stop,
                                               xy_window,
                                               xy_overlap]
                            )

        scale = (64/(64*(1.5 ** 2)), 64/(64*(1.5 ** 2)))
        x_start_stop=[None, None]
        y_start_stop=[None, None]
        xy_window=(64, 64)
        xy_overlap=(1 - 1/8, 1 - 1/8)
        y_start_stop[0] = int (img.shape[0] / 3 + 128)
        y_start_stop[0] = int (img.shape[0] / 2 + 32)

        y_start_stop[1] = y_start_stop[0] + int(2 * (img.shape[0] - y_start_stop[0]) / 3)
        y_start_stop[1] = min(y_start_stop[1], img.shape[0])

        multiscale_cfg.append( ["scale_2", scale,
                                               x_start_stop,
                                               y_start_stop,
                                               xy_window,
                                               xy_overlap]
                            )

        scale = (64/(64 * (1.5 ** 3)), 64/( 64 * (1.5 ** 3)))
        x_start_stop=[None, None]
        y_start_stop=[None, None]
        xy_window=(64, 64)
        xy_overlap=(1 - 1/8, 1 - 1/8)

        y_start_stop[0] = int (img.shape[0] / 2 + 64)
        y_start_stop[1] = y_start_stop[0] + int(2 * (img.shape[0] - y_start_stop[0]) / 3)
        y_start_stop[1] = min(y_start_stop[1], img.shape[0])
        y_start_stop[1] = None

        multiscale_cfg.append( ["scale_3", scale,
                                               x_start_stop,
                                               y_start_stop,
                                               xy_window,
                                               xy_overlap]
                             )


        scale = (64/(64 * (1.5 ** 4)), 64/( 64 * (1.5 ** 4)))
        x_start_stop=[None, None]
        y_start_stop=[None, None]
        xy_window=(64, 64)
        xy_overlap=(1 - 1/8, 1 - 1/8)

        y_start_stop[0] = int (img.shape[0] / 2 + 64)
        #y_start_stop[1] = y_start_stop[0] + int(2 * (img.shape[0] - y_start_stop[0]) / 3)
        #y_start_stop[1] = min(y_start_stop[1], img.shape[0])
        y_start_stop[1] = None
        multiscale_cfg.append(["scale_4", scale,
                                               x_start_stop,
                                               y_start_stop,
                                               xy_window,
                                               xy_overlap]
                             )


        scale = (64/(64 * 0.5), 64/( 64 * 0.5))
        x_start_stop=[None, None]
        y_start_stop=[None, None]
        xy_window=(64, 64)
        xy_overlap=(2/8, 2/8)

        y_start_stop[0] = int (img.shape[0] / 2)
        y_start_stop[1] = y_start_stop[0] + int(1 * (img.shape[0] - y_start_stop[0]) / 8)
        y_start_stop[1] = min(y_start_stop[1], img.shape[0])


        multiscale_cfg.append( ["scale_5", scale,
                                               x_start_stop,
                                               y_start_stop,
                                               xy_window,
                                               xy_overlap]
                            )
        return multiscale_cfg


def apply_threshold(heatmap, threshold):

    """
    Thresholding of heat map image
    """
    heatmap_cpy = np.copy(heatmap)
    # Zero out pixels below the threshold
    heatmap_cpy[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap_cpy

def generate_heat_map(img, detect_wins, threshold = 3):
    """
    Generate heat maps based on detected windows
    """

    zeros = np.zeros_like(img[:,:,0]).astype(np.float64)
    heat_map = np.zeros_like(img[:,:,0]).astype(np.float64)
    for win in detect_wins:
        start_cord = win[0]
        end_cord = win[1]

        start_x = start_cord[0]
        start_y = start_cord[1]
        end_x = end_cord[0]
        end_y = end_cord[1]
        heat_map[start_y:end_y, start_x:end_x] += 1

    thresholded_heat_map = apply_threshold(heat_map, threshold)
    color_theshold_img = np.dstack((zeros, zeros, thresholded_heat_map))
    #cv2.imshow("heat_map", color_theshold_img)
    #cv2.imshow("thresholded_heat_map", thresholded_heat_map)
    return thresholded_heat_map, heat_map

def draw_labeled_bboxes(img, labels):
    """
    Draw the bounding boxes on the object of interest
    once non-maximum supression is performed
    """
    img_cpy = np.copy(img)
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
        cv2.rectangle(img_cpy, bbox[0], bbox[1], (0,255,0), 4)
    # Return the image
    return img_cpy


def test_img_boxes(img):

    """
    code for testing function within the file
    """

    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    #print(windows)

    window_img = draw_boxes(img, windows, color=(0, 255, 0), thick=6)
    cv2.imshow("window_img", window_img)

    scaled_x = img.shape[1]//2
    scaled_y = img.shape[0]//2
    print(scaled_x, scaled_y)
    img_scaled = cv2.resize(img, (scaled_x, scaled_y))
    print(img_scaled.shape)
    windows = slide_window(img_scaled, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))

    #print(windows)
    window_img = draw_boxes(img_scaled, windows, color=(0, 255, 0), thick=6)
    cv2.imshow("window_img_scale", window_img)

def main(argv):
    img_path = argv[1]
    classifier_file = argv[2]

    img = cv2.imread(img_path)

    print("loading classifier from prev trained")
    cl_file = open(classifier_file, "rb")
    classifier = pickle.load(cl_file)
    cl_file.close()
    print(classifier)

    svc_classify = classifier["svc"]
    norm = classifier["norm"]
    print(norm.mean_, norm.scale_, norm.var_)


    features = features_vec()
    # this should be same as used in training

    features.spatial_features = True
    features.hist_features = True
    features.hog_features = True

    features.spatial_cs = "HLS"
    features.hist_cs = "HLS"
    features.hog_cs = "HLS"


    multi_scale_cfg = get_multi_scale_config(img)
    print(multi_scale_cfg)

    test_slide_window(img, multi_scale_cfg)

    multi_scale_wins, win_img = detect_object_multiscale(img, svc_classify, norm, features, multi_scale_cfg, fast_hog_extract = 1)

    thresh_heat_map_img, actual_heat_map = generate_heat_map(img, multi_scale_wins, threshold=3)

    heat_map_img = thresh_heat_map_img
    labels = label(heat_map_img)

    final_img = draw_labeled_bboxes(img, labels)
    cv2.imshow("final_img", final_img)



if __name__ == "__main__":
    main(sys.argv)
    while 1:
        pass
