
import numpy as np
import os
import sys
import cv2

from skimage.feature import hog
import matplotlib.pyplot as plt


# Define a function to compute color histogram features
def color_hist(img, channel = "ALL", nbins=32, bins_range=(0, 256)):

    """
    Obtain color histogram per channel of the image
    """
    # Compute the histogram of the RGB channels separately
    ch0_hist = None
    ch1_hist = None
    ch2_hist = None

    if(channel == "ALL"):
        ch0_hist = np.histogram( img[:,:,0], nbins, bins_range)
        ch1_hist = np.histogram( img[:,:,1], nbins, bins_range)
        ch2_hist = np.histogram( img[:,:,2], nbins, bins_range)
    else:
        ch0_hist = np.histogram( img[:,:, channel], nbins, bins_range)

    # Generating bin centers
    bin_centers = None
    bin_centers = ch0_hist[1]
    bin_centers = (bin_centers[1:] + bin_centers[0:len(bin_centers) - 1])/2

    # Concatenate the histograms into a single feature vector
    hist_features = None

    if(channel == "ALL"):
        hist_features = np.concatenate((ch0_hist[0], ch1_hist[0], ch2_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        #return ch0_hist, ch1_hist, ch2_hist, bin_centers, hist_features
    else:
        hist_features = ch0_hist[0]
        #return ch0_hist, bin_centers, hist_features

    return hist_features
    #return ch0_hist, ch1_hist, ch2_hist, bin_centers, hist_features

# HoG features
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    generate HoG transform of the image channel
    """
    hog_image = None
    #print(orient, pix_per_cell, cell_per_block, vis, feature_vec)
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        #features = [] # Remove this line
        #hog_image = img # Remove this line
        features, hog_image = hog(img, orient, (pix_per_cell, pix_per_cell),
        (cell_per_block, cell_per_block), visualise = vis, feature_vector = feature_vec)
    else:
        # Use skimage.hog() to get features only
        #features = [] # Remove this line
        features = hog(img, orient,
        (pix_per_cell, pix_per_cell), (cell_per_block, cell_per_block), visualise = vis, feature_vector = feature_vec)

    return features, hog_image

class HoG_feature(object):
    def __init__(self):
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.visualize = False
        self.feature_vector = True
        self.channel = "ALL"


def extract_HoG_fvector(img, hog_cfg):

    """
    Extract HoG feature vectors for all the channels of the image
    """
    assert (hog_cfg != None)

    ch0_features = None
    ch1_features = None
    ch2_features = None

    if(hog_cfg.channel == "ALL"):

        #og_cfg.visualize = True
        ch0_features, ch0_hog_img = get_hog_features(img[:,:,0], hog_cfg.orient, hog_cfg.pix_per_cell,
            hog_cfg.cell_per_block, vis=hog_cfg.visualize, feature_vec=hog_cfg.feature_vector)

        ch1_features, ch1_hog_img = get_hog_features(img[:,:,1], hog_cfg.orient, hog_cfg.pix_per_cell,
            hog_cfg.cell_per_block, vis=hog_cfg.visualize, feature_vec=hog_cfg.feature_vector)

        ch2_features, ch2_hog_img = get_hog_features(img[:,:,2], hog_cfg.orient, hog_cfg.pix_per_cell,
            hog_cfg.cell_per_block, vis=hog_cfg.visualize, feature_vec=hog_cfg.feature_vector)

        #uncomment below code to dump hog image
        #cv2.imshow("ch0_hog_img", ch0_hog_img/np.max(ch0_hog_img))
        #cv2.imshow("ch1_hog_img", ch1_hog_img/np.max(ch1_hog_img))
        #cv2.imshow("ch2_hog_img", ch2_hog_img/np.max(ch2_hog_img))

        #cv2.imwrite("./Big-car_ch0_hog_img1" + ".jpg", np.uint8(255 * ch0_hog_img/np.max(ch0_hog_img) ))
        #cv2.imwrite("./Big-car_ch1_hog_img1"+ ".jpg", np.uint8(255 * ch1_hog_img/np.max(ch1_hog_img)) )
        #cv2.imwrite("./Big-car_ch2_hog_img1"+ ".jpg", np.uint8(255 * ch2_hog_img/np.max(ch2_hog_img)))
        #return (ch0_features , ch1_features, ch2_features)
    else:

        ch0_features = get_hog_features(img[:,:, hog_cfg.channel], hog_cfg.orient, hog_cfg.pix_per_cell,
            hog_cfg.cell_per_block, vis=hog_cfg.visualize, feature_vec=hog_cfg.feature_vector)


    return (ch0_features , ch1_features, ch2_features)


        #return   ch0_features, None, None

#spatial feature vectors

def convert_color_space(img, color_space):
    """
    Color Space conversion of the image
    """

    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            #print("convert HLS")
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif color_space == 'RGB':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print("!!! Not supported format")
    else: feature_image = np.copy(img)
    return feature_image

def bin_spatial(img, color_space='BGR', size=(32, 32), ch = "ALL"):
    """
    Spatial features of the image by performing binning operation
    """
    # Convert image to new color space (if specified)
    feature_image = convert_color_space(img, color_space)
    if(ch == "ALL"):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(feature_image, size).ravel()
    else:
        resized_img = cv2.resize(feature_image, size)
        features =  resized_img[:,:,ch].ravel()
    # Return the feature vector
    return features


class features_vec(object):
    def __init__(self):
        self.spatial_features = True
        self.spatial_cs = "RGB"
        self.spatial_ch = "ALL"

        self.hist_features = True
        self.hist_cs = "RGB"
        self.hist_ch = "ALL"
        self.hist_nbins=32
        self.hist_bins_range=(0, 256)

        self.hog_features = True
        self.hog_cs = "RGB"
        self.hog = HoG_feature()


def extract_img_features(img, fvec_cfg ):
    """
    Api to extract image features like bin-spatial, color histogram, HoG features
    """

    #get spatial feature vector
    features = []
    if(fvec_cfg.spatial_features == True):
        #print("spatial_features")
        spatial_features = bin_spatial(img, fvec_cfg.spatial_cs, size = (32,32), ch=fvec_cfg.spatial_ch)
        features.append(spatial_features)

    if(fvec_cfg.hist_features == True):
        #print("hist_features")
        feature_image = convert_color_space(img, fvec_cfg.hist_cs)
        #def color_hist(img, channel = "ALL", nbins=32, bins_range=(0, 256)):
        hist_features = color_hist(feature_image, channel=fvec_cfg.hist_ch,
                                   nbins=fvec_cfg.hist_nbins, bins_range = fvec_cfg.hist_bins_range)
        features.append(hist_features)

    if(fvec_cfg.hog_features == True):

        #print("Hog feature")
        feature_image = convert_color_space(img, fvec_cfg.hog_cs)
        #extract_HoG_fvector(img, hog_cfg):
        fvec_cfg.hog.feature_vector = True
        ch0_features, ch1_features, ch2_features = extract_HoG_fvector(feature_image, fvec_cfg.hog)
        hog_features = np.concatenate((ch0_features, ch1_features, ch2_features))
        features.append(hog_features)

    out_features = None
    if len(features) > 0:
        out_features = np.concatenate(features)
        #print("!!!",len(features), out_features.shape)
    return out_features


def extract_hog_features(hog_img_ch, start_pt = (0, 0), end_pt = (64, 64), pix_per_cell=(8, 8), cells_per_blk = (2, 2)):

    #print(hog_img_ch.shape)
    start_x = start_pt[0]
    start_y = start_pt[1]

    end_x = end_pt[0]
    end_y = end_pt[1]

    start_x_blk_idx = start_x // 8
    start_y_blk_idx = start_y // 8

    end_x_blk_idx = end_x // 8
    end_y_blk_idx = end_y // 8

    #print(start_x_blk_idx, start_y_blk_idx, end_x_blk_idx, end_y_blk_idx, start_pt, end_pt)
    hog_feature = hog_img_ch[start_y_blk_idx:end_y_blk_idx-1, start_x_blk_idx:end_x_blk_idx-1].ravel()
    #print(hog_feature.shape)
    return hog_feature

def plot_hist(img):
    pass

def main(argv):
    image_file = argv[1]

    img = cv2.imread(image_file)

    print(img.shape)

    hog = HoG_feature()
    hog.feature_vector = False
    hog.channel = "ALL"

    ch0_features, ch1_features, ch2_features = extract_HoG_fvector(img, hog)
    print(ch0_features.shape)
    print(ch0_features.shape)
    print(ch0_features.shape)

    features = features_vec()
    features.spatial_features = True
    features.hist_features = True
    features.hog_features = True

    features.spatial_cs = "HLS"
    features.hist_cs = "HLS"
    features.hog_cs = "HLS"
    #features.hist_cs = "HSV"


    output_fvec = extract_img_features(img, features)

    print(output_fvec.shape)



if __name__ == "__main__":
    main(sys.argv)
    #plt.show()
    while 1:
        pass
