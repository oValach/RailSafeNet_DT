import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ultralyticsplus import YOLO, render_result
from test import load, process

PATH_jpgs = 'RailNet_DT/rs19_val/jpgs/test'
PATH_model_seg = 'RailNet_DT/models/modelchp_85_100_0.0002865237576874738_2_0.606629.pth'
PATH_model_det = 'ultralyticsplus/yolov8s'

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

def find_edges(arr, y_levels, values=[0, 1, 6], min_width=19):
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
                filtered_edges = [(start, end) for start, end in filtered_edges if 0 not in (start, end) and 1919 not in (start, end)]
                
                edges_dict[y] = filtered_edges

        edges_dict = {k: v for k, v in edges_dict.items() if v}
        
        return edges_dict

def mark_edges(arr, edges_dict, mark_value=30):
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

def find_rail_sides(edges_dict):
        left_border = []
        right_border = []
        for y,xs in edges_dict.items():
                left_border.append([min(xs)[0],y])
                right_border.append([max(xs)[1],y])

        # funkce outlieru zastavi na prvni nespojitosti -> delsi zona mela nespojitost na konci -> chci tu
        left_border = robust_rail_sides(left_border) # filter outliers
        
        right_border = robust_rail_sides(right_border)
        
        return left_border, right_border

def robust_rail_sides(border, threshold=1.5):
        border = np.array(border)
        
        steps_x = np.diff(border[:, 0])
        median_step = np.median(steps_x)
        
        threshold_step = np.abs(threshold*np.abs(median_step))
        treshold_overcommings = abs(steps_x) > abs(threshold_step)
        
        if True not in treshold_overcommings:
                return border
        else:
                overcommings_indices = [i for i, element in enumerate(treshold_overcommings) if element == True]
                filtered_border = border
                
                previously_deleted = []
                for i in overcommings_indices:
                        for item in previously_deleted:
                                if item[0] < i:
                                        i -= item[1]
                        left_border = filtered_border[i+1:]
                        right_border = filtered_border[:i+1]
                        if len(right_border)<2:
                                filtered_border = left_border
                                previously_deleted.append([i,len(right_border)])
                        elif len(left_border)<2:
                                filtered_border = right_border
                                previously_deleted.append([i,len(left_border)])
                        else:
                                filtered_border = np.concatenate((robust_rail_sides(right_border),robust_rail_sides(left_border)), axis=0)
                
                return filtered_border

def find_dist_from_edges(image, edges_dict, left_border, right_border, real_life_width_mm, real_life_target_mm, mark_value=30):
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
        average_diffs = {k: sum(e-s for s, e in v) / len(v) for k, v in edges_dict.items() if v}
        # Pixel to mm scale factor
        scale_factors = {k: real_life_width_mm / v for k, v in average_diffs.items()}
        # Converting the real-life target distance to pixels
        target_distances_px = {k: int(real_life_target_mm / v) for k, v in scale_factors.items()}

        mark=1
        
        # Mark the regions representing the target distance to the left and right from the furthest edges
        end_points_left = {}
        for point in left_border:
                min_edge = point[0]
                
                # Ensure we stay within the image bounds
                left_mark_start = max(0, min_edge - int(target_distances_px[point[1]]))
                if left_mark_start != 0:
                        end_points_left[point[1]] = left_mark_start
                
                if mark:
                        # Mark the left region
                        if left_mark_start < min_edge:
                                image[point[1], left_mark_start:min_edge] = mark_value
        
        end_points_right = {}
        for point in right_border:
                max_edge = point[0]
                
                # Ensure we stay within the image bounds
                right_mark_end = min(image.shape[1], max_edge + int(target_distances_px[point[1]]))
                if right_mark_end != image.shape[1]:
                        end_points_right[point[1]] = right_mark_end
                
                if mark:
                        # Mark the right region
                        if max_edge < right_mark_end:
                                image[point[1], max_edge:right_mark_end] = mark_value

        return image, end_points_left, end_points_right

def bresenham_line(x0, y0, x1, y1):
        """
        Generate the coordinates of a line from (x0, y0) to (x1, y1) using Bresenham's algorithm.
        """
        line = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy  # error value e_xy

        while True:
                line.append((x0, y0))  # Add the current point to the line
                if x0 == x1 and y0 == y1:
                        break
                e2 = 2 * err
                if e2 >= dy:  # e_xy+e_x > 0
                        err += dy
                        x0 += sx
                if e2 <= dx:  # e_xy+e_y < 0
                        err += dx
                        y0 += sy

        return line

def interpolate_end_points(end_points_dict):
        line_arr = []
        ys = list(end_points_dict.keys())
        xs = list(end_points_dict.values())

        for i in range(0, len(ys) - 1):
                y1, y2 = ys[i], ys[i + 1]
                x1, x2 = xs[i], xs[i + 1]
                line = bresenham_line(x1, y1, x2, y2)
                line_arr = line_arr + line
                
        return line_arr

def extrapolate_line(pixels, image, min_y=None):
        """
        Extrapolate a line based on the last segment using linear regression.

        Parameters:
        - pixels: List of (x, y) tuples representing line pixel coordinates.
        - image: 2D numpy array representing the image.
        - max_y: Maximum y-value to extrapolate to (optional).

        Returns:
        - A list of new extrapolated (x, y) pixel coordinates.
        """
        # Check if the pixel list is shorter than the window for regression
        if len(pixels) < 10:
                raise ValueError("Not enough pixels to perform extrapolation.")

        # Take the last 30 pixels for the regression
        recent_pixels = np.array(pixels[-30:])
        
        # Prepare data for regression
        X = recent_pixels[:, 0].reshape(-1, 1)  # Reshape for sklearn
        y = recent_pixels[:, 1]

        # Fit the linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Get the coefficients
        slope = model.coef_[0]
        intercept = model.intercept_

        # Define the extrapolation function
        extrapolate = lambda x: slope * x + intercept

        # Initialize with the last known pixel
        last_pixel = pixels[-1]
        new_pixels = []
        x, y = last_pixel

        # Calculate the direction for extrapolation
        dx = 1 if pixels[-1][0] - pixels[-2][0] > 0 else -1

        # Minimum y limit is either the provided min_y or the height of the image
        min_y = min_y if min_y is not None else image.shape[0] - 1

        # Extrapolate until we hit the min y limit or the border of the image
        while 0 <= x < image.shape[1] and min_y < y < image.shape[0]:
                x += dx
                y = int(extrapolate(x))
                
                # Check bounds
                if 0 <= y < image.shape[0]:
                        if 0 <= x < image.shape[1]:
                                new_pixels.append((x, y))
                else:
                        break  # Stop if we go outside the image bounds

        return new_pixels

def extrapolate_borders(dist_marked_id_map, border_l, border_r, lowest_y):
        
        border_extrapolation_l1 = extrapolate_line(border_l, dist_marked_id_map, lowest_y)
        border_extrapolation_l2 = extrapolate_line(border_l[::-1], dist_marked_id_map, lowest_y)
        
        border_l = border_l + border_extrapolation_l1 + border_extrapolation_l2
        
        border_extrapolation_r1 = extrapolate_line(border_r, dist_marked_id_map, lowest_y)
        border_extrapolation_r2 = extrapolate_line(border_r[::-1], dist_marked_id_map, lowest_y)
        
        border_r = border_r + border_extrapolation_r1 + border_extrapolation_r2
        
        return border_l, border_r

def find_zone_border(image, edges, irl_width_mm=1435, irl_target_mm=1000, lowest_y = 0):
        
        irl_width_mm = 1435
        
        left_border, right_border = find_rail_sides(edges)
        
        dist_marked_id_map, end_points_left, end_points_right = find_dist_from_edges(image, edges, left_border, right_border, irl_width_mm, irl_target_mm+70) # 1 meter + 70mm rail width
        
        border_l = interpolate_end_points(end_points_left)
        border_r = interpolate_end_points(end_points_right)
        
        border_l, border_r = extrapolate_borders(dist_marked_id_map, border_l, border_r, lowest_y)
        
        return [border_l, border_r]

def visualize(id_map, borders):
        for border in borders:
                for point in border[0]:
                        id_map[point[1],point[0]] = 30
                for point in border[1]:
                        id_map[point[1],point[0]] = 30
        plt.imshow(id_map)
        plt.show()

def get_clues(segmentation_mask, number_of_clues):
        
        lowest, highest = find_extreme_y_values(segmentation_mask)
        clue_step = int((highest - lowest) / number_of_clues+1)
        clues = []
        for i in range(number_of_clues):
                clues.append(highest - (i*clue_step))
        
        return clues

def border_handler(id_map, edges, target_distances, vis=False):
        
        lowest, _ = find_extreme_y_values(segmentation_mask)
        borders = []
        for target in target_distances:
                borders.append(find_zone_border(id_map, edges, irl_target_mm=target, lowest_y = lowest))
                
        if vis:
                visualize(id_map, borders)
                
        return borders

def segment(image_size, filename, PATH_jpgs, PATH_model, model_type):
        
        image_norm, image, mask, _, model = load(filename, PATH_jpgs, PATH_model, image_size)
        id_map = process(model, image_norm, mask, model_type)
        id_map = cv2.resize(id_map, [1920,1080], interpolation=cv2.INTER_NEAREST)
        
        return id_map

def detect(PATH_model, filename_img, PATH_jpgs):
        
        model = YOLO(PATH_model)

        model.overrides['conf'] = 0.25  # NMS confidence threshold
        model.overrides['iou'] = 0.45  # NMS IoU threshold
        model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        model.overrides['max_det'] = 1000  # maximum number of detections per image

        image = cv2.imread(os.path.join(PATH_jpgs, filename_img))
        results = model.predict(image)

        return results

def manage_detections(results, model):
        names = model.model.names
        bbox = results[0].boxes.xyxy.tolist()
        cls = results[0].boxes.cls.tolist()
        accepted_stationary = np.array([0,1,2,3,4,5,7,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,56,57,58,59,60,61,62,63,68,69,70,71,72,73,74,75,78,79])
        accepted_moving = np.array([0,1,2,3,4,5,7,15,16,17,18,19,20,21,22,23])
        boxes_moving = {}
        boxes_stationary = {}
        if len(bbox) > 0:
                for xyxy, clss in zip(bbox, cls):
                        if clss in accepted_moving:
                                if clss in boxes_moving.keys() and len(boxes_moving[clss]) > 0:
                                        boxes_moving[clss] = boxes_moving[clss].append(xyxy)
                                else:
                                        boxes_moving[clss] = xyxy
                        if clss in accepted_stationary:
                                if clss in boxes_stationary.keys() and len(boxes_stationary[clss]) > 0:
                                        boxes_stationary[clss] = boxes_stationary[clss].append(xyxy)
                                else:
                                        boxes_stationary[clss] = xyxy

vis = 1

for filename_img in os.listdir(PATH_jpgs):
        
        # Segmentation
        image_size = [1024,1024]
        model_type = "segformer" #deeplab
        segmentation_mask = segment(image_size, filename_img, PATH_jpgs, PATH_model_seg, model_type)

        # Border search
        clues = get_clues(segmentation_mask, 10)
        edges = find_edges(segmentation_mask, clues, min_width=int(segmentation_mask.shape[1]*0.015))
        #id_map_marked = mark_edges(segmentation_mask, edges)
        
        target_distances = [1000,2000,3000]
        borders = border_handler(segmentation_mask, edges, target_distances, vis=True)
        
        # Detection
        results, model = detect(PATH_model_det, filename_img, PATH_jpgs)
        detections = manage_detections(results, model)
        