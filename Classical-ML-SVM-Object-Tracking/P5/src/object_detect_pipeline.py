import numpy as np
import cv2
import sys
import pickle

from sklearn.preprocessing import StandardScaler
import time
from sklearn.svm import LinearSVC

from object_detect import *
import time

from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip


def object_detect(classifier_file, img_file=None, video_file=None, save_file = None, dump_file=None):

    """
    Main function to perform object detection and localization on image or video files
    """

    assert (classifier_file != None)
    print("SVM-classifier:{0:}".format(classifier_file), "Image File: {0:}".format(img_file),
            "Video File: {0:}".format(video_file), "Save File/path:{0:}".format(save_file))
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

    # process Image file
    if img_file != None:

        for i in range(1): # run the detection multiple times on same frame just to debug
            img = cv2.imread(img_file)
            multi_scale_cfg = get_multi_scale_config(img)

            detected_out_img, thresh_heat_map, orgi_heat_map, img_multiscale_detect, detected_wins = vehicle_detect_pipeline(img, svc_classify,
                                                                            norm, features, multi_scale_cfg, fast_hog_extract = 1, draw_win_flag = 1)


            #different representation of heat map for visualizations
            heat_map = orgi_heat_map # it is non threshholded version
            color_heat_img = np.dstack((np.zeros_like(heat_map), np.zeros_like(heat_map), orgi_heat_map))
            if(np.max(heat_map) != 0):
                color_heat_img_1 = np.dstack((heat_map/np.max(heat_map), heat_map/np.max(heat_map), heat_map/np.max(heat_map))) # gray scale for different intensities
                thresh_color_heat_img_1 = np.dstack((thresh_heat_map/np.max(thresh_heat_map),
                                                    thresh_heat_map/np.max(thresh_heat_map), thresh_heat_map/np.max(thresh_heat_map))) # gray scale for different intensities

            else:
                color_heat_img_1 = np.dstack((heat_map, heat_map, heat_map)) # gray scale for different intensities
                thresh_color_heat_img_1 = np.dstack((thresh_heat_map,
                                                    thresh_heat_map, thresh_heat_map)) # gray scale for different intensities

            heat_map_hot = cv2.applyColorMap(np.uint8(color_heat_img_1 * 255), cv2.COLORMAP_HOT)
            thresh_heat_map_hot = cv2.applyColorMap(np.uint8(thresh_color_heat_img_1 * 255), cv2.COLORMAP_HOT)

            print("-----------------+++",i)
            cv2.imshow("detected_out_img_{0:}".format(i), detected_out_img)
            #cv2.imshow("heat_map_{0:}".format(i), heat_map)
            #cv2.imshow("color_heat_map_{0:}".format(i), color_heat_img)
            #cv2.imshow("color_heat_map_{0:}".format(i), color_heat_img / np.max(color_heat_img[:,:,2]))
            cv2.imshow("heat_map_hot_{0:}".format(i), heat_map_hot)
            cv2.imshow("thresholded_heat_map_hot_{0:}".format(i), thresh_heat_map_hot)
            cv2.imshow("img_multiscale_detect_{0:}".format(i), img_multiscale_detect)

            if save_file != None:
                #save the output
                cv2.imwrite(save_file + "detected_out_img_{0:}".format(i) + ".png", detected_out_img)
                #cv2.imwrite(save_file + "color_heat_map_{0:}".format(i) + ".jpg", np.uint8((color_heat_img / np.max(color_heat_img[:,:,2]) * 255 )))
                cv2.imwrite(save_file + "heat_map_hot{0:}".format(i) + ".png",heat_map_hot)
                cv2.imwrite(save_file + "thresholded_heat_map_hot{0:}".format(i) + ".png",thresh_heat_map_hot)
                #cv2.imshow(save_file + "color_heat_map_{0:}".format(i) + ".jpg", color_heat_img / np.max(color_heat_img[:,:,2]))
                cv2.imwrite(save_file + "img_multiscale_detect_{0:}".format(i) + ".png", img_multiscale_detect)



    #process Video file
    if video_file != None:
        #process video
        #pass
        #read vido file
        video = VideoFileClip(video_file)
        video = video.subclip(0)
        out_frames = []
        frame_cnt = 0
        in_frames = []

        #img = video.get_frame(0)
        #multi_scale_cfg = get_multi_scale_config(img)
        #print(multi_scale_cfg, img.shape)

        #process each frame
        heat_maps = []
        for frame in video.iter_frames(progress_bar=True):
            #frame = video.get_frame(i)
            #print("next frame")
            multi_scale_cfg = get_multi_scale_config(frame)
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("in_frame", img)
            in_frames.append(frame)
            print(frame.dtype)
            out_frame, frame_heat_map, orig_frame_heat_map, frame_multiscale_detect, detected_wins_frame = vehicle_detect_pipeline(img, svc_classify,
                                                                                        norm, features, multi_scale_cfg, fast_hog_extract = 1, draw_win_flag=0,
                                                                                        heat_map_list = heat_maps)

            out_frames.append(out_frame[:, :, [2, 1, 0]])

            cv2.imshow("output_frame", out_frame)
            cv2.waitKey(10)
            frame_cnt += 1
            if dump_file != None:
                print(dump_file + "/rkb_{0:}".format(frame_cnt))
                cv2.imwrite(dump_file + "/rkb_{0:}.png".format(frame_cnt), img)
            #dump the frame
            #break

        print("End of clip")
        new_clip = ImageSequenceClip(out_frames, fps=video.fps)
        if save_file == None:
            save = "new_file_1.mp4"
        new_clip.write_videofile(save_file)
        del new_clip
        del video




def predict_heat_img(heat_map_list, frame_heat_map, max_depth = 3):
    """
    Function to predict heat images for successive video frame based on the previous frames' heat image
    """

    heat_img = np.copy(frame_heat_map)

    count  = 1
    for heat_map in  heat_map_list:
        heat_img = np.add(heat_map, heat_img)
        #cv2.imshow("count{0:}".format(count), heat_map)
        count += 1
        #print(np.max(heat_img), np.max(heat_map))


    heat_img = np.clip(heat_img, 0, 255)
    threshold = 10
    thresholded_heat_map = apply_threshold(heat_img, threshold)

    #print(len(heat_map_list))
    if len(heat_map_list) < max_depth:
        heat_map_list.append(frame_heat_map)
        #cv2.imshow("heat_map_list", heat_map_list[-2])
    else:
        del  heat_map_list[0]
        #print(len(heat_map_list))
        heat_map_list.append(frame_heat_map)
        #cv2.imshow("heat_map_list", heat_map_list[-2])

    #cv2.imshow("frame_heat_map", frame_heat_map)
    #cv2.imshow("thresholded_pred_heat_map", thresholded_heat_map)
    return thresholded_heat_map

def histogram_equalization(img):

    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_hls[:,:,1] = cv2.equalizeHist(img_hls[:,:,1])
    out_img = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR)
    cv2.imshow("hist_eq", out_img)
    return out_img


def vehicle_detect_pipeline(img, classifier, normalizer, features, multi_scale_cfg, fast_hog_extract = 1, draw_win_flag = 1, heat_map_list=None):

    """
    Pipeline to detect and localize cars in the image frame
    """
    svc_classify = classifier
    norm = normalizer

    multi_scale_wins, win_img = detect_object_multiscale(img, svc_classify, norm, features, multi_scale_cfg, fast_hog_extract, draw_win_flag)

    thresh_heat_map_img, ori_heat_map = generate_heat_map(img, multi_scale_wins, threshold=3)

    #color_theshold_img = np.dstack((zeros, zeros, heat_map_img))
    #cv2.imshow("heat_map", color_theshold_img)

    heat_map_img = thresh_heat_map_img
    if heat_map_list != None:
        predict_heat_map = predict_heat_img(heat_map_list, heat_map_img, max_depth=5)
        #print("---", len(heat_map_list))
        #cv2.imshow("heat-list", heat_map_list[-1])
    else:
        predict_heat_map = heat_map_img

    labels = label(predict_heat_map)
    output_img = draw_labeled_bboxes(img, labels)

    return output_img, predict_heat_map, ori_heat_map, win_img, multi_scale_wins
    #cv2.imshow("final_img", final_img)

def main(argv):

    classifier_file = argv[1]
    video_flag = int(argv[2])

    image_file = None #image file name
    video_file = None # video file name
    save_file = None  # output file name/path
    dump_file = None  # path to dump intermediate frames output in video
    if video_flag == 1:
        video_file = argv[3]
        save_file = argv[4]
        #dump_file = argv[5]

    else:
        image_file = argv[3]
        save_file = argv[4]

    object_detect(classifier_file, image_file, video_file, save_file, dump_file)

if __name__ == "__main__":

    main(sys.argv)
    while 1:
        pass
