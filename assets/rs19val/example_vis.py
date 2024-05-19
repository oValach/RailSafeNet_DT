# Copyright Oliver Zendel, AIT Austrian Institute of Technology Gmbh 2019
# Example visualization tool to illustrate interpretation of RailSem19 data usage

import sys, os, glob,  math, argparse
import numpy as np
import cv2, json
import matplotlib.pyplot as plt

rs19_label2bgr = {"buffer-stop": (70,70,70),
                "crossing": (128,64,128),
                "guard-rail": (0,255,0),
                "train-car" :  (100,80,0),
                "platform" : (232,35,244),
                "rail": (255,255,0),
                "switch-indicator": (127,255,0),
                "switch-left": (255,255,0),
                "switch-right": (127,127,0),
                "switch-unknown": (191,191,0),
                "switch-static": (0,255,127),
                "track-sign-front" : (0,220,220),
                "track-signal-front" : (30,170,250),
                "track-signal-back" : (0,85,125),
                #rail occluders
                "person-group" : (60,20,220),
                "car" : (142,0,0),
                "fence" : (153,153,190),
                "person" : (60,20,220),
                "pole" : (153,153,153),
                "rail-occluder" : (255,255,255),
                "truck" : (70,0,0)
                }

def files_in_subdirs(start_dir, pattern = ["*.png","*.jpg","*.jpeg"]):
    files = []
    for p in pattern:
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob.glob(os.path.join(dir,p)))
    return files

def config_to_rgb(inp_path_config_json, default_col = [255,255,255]):
    lut = []
    inp_json = json.load(open(inp_path_config_json, 'r'))
    for c in range(3): #for each color channel
        lut_c =[l["color"][c] for l in inp_json["labels"]]+[default_col[c]]*(256-len(inp_json["labels"]))
        lut.append(np.asarray(lut_c, dtype=np.uint8))
    return lut

def corss_hatch_rail(im_vis, coords, color_left=(255,255,0), color_right=(127,127,0)):
    ml = min(len(coords[0]), len(coords[1]))
    for i in range(ml):
        midpnt = ((coords[0][i][0]+coords[1][i][0])//2, (coords[0][i][1]+coords[1][i][1])//2)
        cv2.line(im_vis, tuple(coords[0][i]), midpnt, color_left)
        cv2.line(im_vis, midpnt, tuple(coords[1][i]), color_right)

def json_to_img(inp_path_json, line_thickness=2):
    inp_json = json.load(open(inp_path_json, 'r'))
    im_json = np.zeros((inp_json["imgHeight"], inp_json["imgWidth"],3), dtype=np.uint8)
    for obj in inp_json["objects"]:
        col = rs19_label2bgr.get(obj["label"],[255,255,255])
        if "boundingbox" in obj:
            cv2.rectangle(im_json, tuple(obj["boundingbox"][0:2]), tuple(obj["boundingbox"][2:4]), col, line_thickness)
        elif "polygon" in obj:
            pnts_draw = np.around(np.array(obj["polygon"])).astype(np.int32)
            cv2.polylines(im_json, [pnts_draw], True, col, line_thickness)
        elif "polyline-pair" in obj:
            #left rail of a rail pair has index 0, right rail of a rail pair has index 1
            rails_draw = [np.around(np.array(obj["polyline-pair"][i])).astype(np.int32) for i in range(2)]
            corss_hatch_rail(im_json, obj["polyline-pair"],  rs19_label2bgr['switch-left'], rs19_label2bgr['switch-right'])
            cv2.polylines(im_json, rails_draw, False, col)
        elif "polyline" in obj:
            rail_draw = np.around(np.array(obj["polyline"])).astype(np.int32)
            cv2.polylines(im_json, [rail_draw], False, col, line_thickness)
    return im_json, inp_json["frame"]

def get_joined_img(inp_path_json, jpg_folder, uint8_folder, lut_bgr, blend_vals=[0.65,0.25,0.1]):
    im_json, frameid = json_to_img(inp_path_json, line_thickness=2) #visualize geometric annotations
    inp_path_jpg = os.path.join(jpg_folder,frameid+".jpg")
    inp_path_uint8 = os.path.join(uint8_folder,frameid+".png")
    im_jpg = cv2.imread(inp_path_jpg) #add intensity image as background
    im_id_map = cv2.imread(inp_path_uint8,cv2.IMREAD_GRAYSCALE) #get semantic label map
    im_id_col = np.zeros((im_id_map.shape[0], im_id_map.shape[1], 3), np.uint8)
    for c in range(3):
        im_id_col[:,:,c] = lut_bgr[c][im_id_map] #apply color coding
    
    #plt.imshow(im_id_col)
    #plt.show()

    blend_sources = (im_jpg*blend_vals[0]+im_id_col*blend_vals[1]+im_json*blend_vals[2]).astype(np.uint8)
    return (blend_sources) #blend all three data sources

def vis_all_json(json_folder, jpg_folder, uint8_folder, inp_path_config_json):
    all_json = files_in_subdirs(json_folder, pattern = ["*.json"])
    lut_bgr = config_to_rgb(inp_path_config_json, default_col = [255,255,255])[::-1] #we need to swap color channels as opencv uses BGR
    curr_idx, retKey = 0, ord('a')
    while retKey > ord(' '):
        im_vis = get_joined_img(all_json[curr_idx], jpg_folder, uint8_folder, lut_bgr)
        cv2.putText(im_vis, all_json[curr_idx],(0,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow("RailSem19 Annotation Visualization (use 'a' or 'd' to navigate, ESC to quit)",im_vis)
        retKey = cv2.waitKey(-1) #use 'a' and 'd' to scroll through frames
        curr_idx = curr_idx - 1 if retKey == ord('a') else curr_idx + 1
        curr_idx = (curr_idx + len(all_json)) % len(all_json) #wrap around
    return 0

def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpgs', type=str, default="RailNet_DT/rs19_val/jpgs/rs19_val",
                        help="Folder containing all RGB intensity images")
    parser.add_argument('--jsons', type=str, default="RailNet_DT/rs19_val/jsons/rs19_val",
                        help="Folder containing all geometry-based annotation files")
    parser.add_argument('--uint8', type=str, default="RailNet_DT/rs19_val/uint8/rs19_val",
                        help="Folder containing all dense semantic segm. id maps")
    parser.add_argument('--config', type=str, default="RailNet_DT/rs19_val/rs19-config.json",
                        help="Path to config json for dense label map interpretation")
    args = parser.parse_args(argv)
    return vis_all_json(args.jsons, args.jpgs, args.uint8, args.config)
    
if __name__ == "__main__":
    sys.exit(main())
