import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sahi.prediction import ObjectPrediction
from ultralyticsplus import YOLO, render_result
from test import load, process

PATH_jpgs = 'RailNet_DT/rs19_val/jpgs/test'
PATH_model = 'RailNet_DT/models/modelchp_85_100_0.0002865237576874738_2_0.606629.pth'


def find_extreme_y_values(arr, values=[0, 6]):
        """
        Optimized function to find the lowest and highest y-values (row indices) in a 2D array where 0 or 6 appears.
        
        Parameters:
        - arr: The input 2D NumPy array.
        - values: The values to search for (default is [0, 6]).
        
        Returns:
        A tuple (lowest_y, highest_y) representing the lowest and highest y-values. If values are not found, returns None.
        """
        mask = np.isin(arr, values)
        rows_with_values = np.any(mask, axis=1)
        
        y_indices = np.nonzero(rows_with_values)[0]  # Directly finding non-zero (True) indices
        
        if y_indices.size == 0:
                return None  # Early return if values not found
        
        return y_indices[0], y_indices[-1]

def find_edges(arr, y_levels, values=[0, 1, 6], min_width=38):
        """
        Find start and end positions of continuous sequences of specified values at given y-levels in a 2D array,
        filtering for sequences that meet or exceed a specified minimum width.

        Parameters:
        - arr: 2D NumPy array to search within.
        - y_levels: List of y-levels (row indices) to examine.
        - values: Values to search for (default is [0, 6]).
        - min_width: Minimum width of sequences to be included in the results.

        Returns:
        A dict with y-levels as keys and lists of (start, end) tuples for each sequence found in that row that meets the width criteria.
        """
        edges_dict = {}
        for y in y_levels:
                row = arr[y, :]
                mask = np.isin(row, values).astype(int)
                padded_mask = np.pad(mask, (1, 1), 'constant', constant_values=0)
                diff = np.diff(padded_mask)
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0] - 1

                # Filter sequences based on the minimum width criteria
                filtered_edges = [(start, end) for start, end in zip(starts, ends) if end - start + 1 >= min_width]

                edges_dict[y] = filtered_edges

        return edges_dict

def mark_edges(arr, edges_dict, mark_value):
        """
        Marks a 5x5 zone around the edges found in the array with a specific value.

        Parameters:
        - arr: The original 2D NumPy array.
        - edges_dict: A dictionary with y-levels as keys and lists of (start, end) tuples for edges.
        - mark_value: The value used to mark the edges.

        Returns:
        The modified array with marked zones.
        """
        marked_arr = np.copy(arr)  # Create a copy of the array to avoid modifying the original
        offset = 2  # To mark a 5x5 area, we go 2 pixels in each direction from the center

        for y, edges in edges_dict.items():
                for start, end in edges:
                        # Mark a 5x5 zone around the start and end positions
                        for dy in range(-offset, offset + 1):
                                for dx in range(-offset, offset + 1):
                                        # Check array bounds before marking
                                        if 0 <= y + dy < marked_arr.shape[0] and 0 <= start + dx < marked_arr.shape[1]:
                                                marked_arr[y + dy, start + dx] = mark_value
                                        if 0 <= y + dy < marked_arr.shape[0] and 0 <= end + dx < marked_arr.shape[1]:
                                                marked_arr[y + dy, end + dx] = mark_value

        return marked_arr

def mark_dist_from_edges(marked_image, edges_dict, real_life_width_mm, real_life_target_mm, mark_value=30):
        """
        Mark regions representing a real-life distance (e.g., 2 meters) to the left and right from the furthest edges.
        
        Parameters:
        - arr: 2D NumPy array representing the image.
        - edges_dict: Dictionary with y-levels as keys and lists of (start, end) tuples for edges.
        - real_life_width_mm: The real-world width in millimeters that the average sequence width represents.
        - real_life_target_mm: The real-world distance in millimeters to mark from the edges.
        
        Returns:
        - A NumPy array with the marked regions.
        """
        # Calculate the average sequence width in pixels
        average_diffs = {k: sum(e-s for s, e in v) / len(v) for k, v in edges_dict.items()}
        # Pixel to mm scale factor
        scale_factors = {k: real_life_width_mm / v for k, v in average_diffs.items()}
        # Converting the real-life target distance to pixels
        target_distances_px = {k: int(real_life_target_mm / v) for k, v in scale_factors.items()}

        # Mark the regions representing the target distance to the left and right from the furthest edges
        for y, edge_list in edges_dict.items():
                min_edge = min(edge_list)[0]
                max_edge = max(edge_list)[1]
                
                # Ensure we stay within the image bounds
                left_mark_start = max(0, min_edge - int(target_distances_px[y]))
                right_mark_end = min(marked_image.shape[1], max_edge + int(target_distances_px[y]))
                
                # Mark the left region
                if left_mark_start < min_edge:
                        marked_image[y, left_mark_start:min_edge] = mark_value
                
                # Mark the right region
                if max_edge < right_mark_end:
                        marked_image[y, max_edge:right_mark_end] = mark_value

        return marked_image

show = 1
for filename in os.listdir(PATH_jpgs):
        
        # Segmentation
        image_size = [1024,1024]
        image_norm, image, mask, _, model = load(filename, PATH_model, image_size)
        model_type = "segformer" #deeplab
        id_map = process(model, image_norm, mask, model_type)
        resized_id_map = cv2.resize(id_map, [1920,1080], interpolation=cv2.INTER_NEAREST)
        
        # Detection
        model = YOLO('ultralyticsplus/yolov8s')

        model.overrides['conf'] = 0.25  # NMS confidence threshold
        model.overrides['iou'] = 0.45  # NMS IoU threshold
        model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        model.overrides['max_det'] = 1000  # maximum number of detections per image

        image = cv2.imread(os.path.join(PATH_jpgs, filename))
        results = model.predict(image)

        names = model.model.names
        bbox = results[0].boxes.xyxy.tolist()
        cls = results[0].boxes.cls.tolist()
        accepted_stationary = np.array([0,1,2,3,4,5,7,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,56,57,58,59,60,61,62,63,68,69,70,71,72,73,74,75,78,79])
        accepted_moving = np.array([0,1,2,3,4,5,7,15,16,17,18,19,20,21,22,23])
        boxes_moving = {}
        boxes_stationary = {}
        if len(bbox) > 0:
                for xyxy, cls in zip(bbox, cls):
                        if cls in accepted_moving:
                                if len(boxes_moving[cls]) > 0:
                                        boxes_moving[cls] = boxes_moving[cls].append(xyxy)
                                else:
                                        boxes_moving[cls] = xyxy
                        if cls in accepted_stationary:
                                if len(boxes_stationary[cls]) > 0:
                                        boxes_stationary[cls] = boxes_stationary[cls].append(xyxy)
                                else:
                                        boxes_stationary[cls] = xyxy

        lowest_y, highest_y = find_extreme_y_values(resized_id_map)
        clue_interval = int((highest_y - lowest_y) / 5)
        clues = [highest_y-4*clue_interval,highest_y-3*clue_interval,highest_y-2*clue_interval, highest_y-clue_interval, highest_y]
        
        edges = find_edges(resized_id_map, clues, min_width=int(resized_id_map.shape[1]*0.02))
        
        id_map_marked = mark_edges(resized_id_map, edges, 30)
        
        dist_marked_id_map = mark_dist_from_edges(id_map_marked, edges, 1435, 2070)
        plt.imshow(dist_marked_id_map)
        plt.show()
        # min number of continuous pixels of interrail zone is 2% of pictures width
        if show:
                render = render_result(model=model, image=image, result=results[0])
                plt.figure(figsize=(5, 2.5))
                plt.subplot(1, 2, 1)
                plt.imshow(id_map)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(render)
                plt.axis('off')
                plt.show()