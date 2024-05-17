import os, glob
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from PIL import Image

translate_dict = {0: "road",
                    1:"sidewalk",
                    2:"construction",
                    3:"tram-track",
                    4:"fence",
                    5:"pole",
                    6:"traffic-light",
                    7:"traffic-sign",
                    8:"vegetation",
                    9:"terrain",
                    10:"sky",
                    11:"human",
                    12:"rail-track",
                    13:"car",
                    14:"truck, bus",
                    15:"trackbed",
                    16:"on-rails",
                    17:"rail-raised",
                    18:"rail-embedded",
                    19:"void"
                }           

def files_in_subdirs(start_dir, pattern = ["*.png","*.jpg","*.jpeg"]):
    files = []
    for p in pattern:
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob.glob(os.path.join(dir,p)))
    return files

def export_objects_json(json_files):
    f_index = 0
    objects = {}
    files = len(json_files)
    for json_dir in json_files:
        object_list = {}
        json_file = open(json_dir)
        data = json.load(json_file)

        for dic in data['objects']:
            if dic['label'] not in object_list:
                object_list[dic['label']] = 1
            else:
                object_list[dic['label']] += 1
        
        objects[f_index] = object_list
        f_index += 1
        if f_index % 100 == 0:
            print(f_index)

    return objects

def export_objects_mask(mask_files, dict, global_pixel_count):
    f_index = 0
    objects = {}
    for mask_dir in mask_files:
        object_list = {}
        with Image.open(mask_dir) as img:
                img_array = np.array(img)
                unique_ids, counts = np.unique(img_array, return_counts=True)
                if 255 in unique_ids:
                    unique_ids[list(unique_ids).index(255)] = 19
                object_list = {dict[uid]: 1 for uid in unique_ids}

                for id,count in zip(unique_ids,counts):
                    if id in global_pixel_count.keys():
                        global_pixel_count[id] += count
                    else:
                        global_pixel_count[id] = count
                        
                if 3 in unique_ids:
                    print(mask_dir)
                    
        objects[f_index] = object_list
        f_index += 1
        if f_index % 100 == 0:
            print(f_index)

    return objects, global_pixel_count

def get_stats(objects_exported):
    all_keys_count = {}
    file_all_keys_count = {}
    file_unique_keys_count = {}
    for file_n, dic in objects_exported.items():
        for key, count in dic.items():
            # Get amount of each label in all files
            if key not in all_keys_count:
                all_keys_count[key] = count
            else:
                all_keys_count[key] += count
        # Get sum of keys in file
        file_all_keys_count[file_n] = sum(dic.values())
        file_unique_keys_count[file_n] = len(dic.keys())
    
    unique_keys = all_keys_count.keys()
    img_count = len(objects_exported)

    key_not_present_count = {}
    for key in unique_keys:
        key_not_present_count[key] = 0

    for file_n, dic in objects_exported.items():
        for key in all_keys_count.keys():
            if key not in dic.keys():
                if key not in key_not_present_count:
                    key_not_present_count[key] = 1
                else:
                    key_not_present_count[key] += 1

    return(all_keys_count,file_all_keys_count,file_unique_keys_count,unique_keys,img_count,key_not_present_count)

def display_hist1(data_dict):
    plt.style.use('bmh')  # Set the style to bmh

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Get the keys and values from the dictionary
    keys = translate_dict.values()
    values = list(data_dict.values())

    # Plot the histogram
    rects = ax.bar(keys, values, color='#8B0000')

    # Add text to each bar
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height+90000000, format_number(values[i]), ha='center', va='center', color='black')

    
    # Set the labels and title
    ax.set_xlabel('Class Name', fontsize=24)
    ax.set_ylabel('Class Volume [px]', fontsize=24)
    ax.set_xticklabels(keys, rotation=50, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)  # Set y-axis tick label font size

    # Show the plot
    plt.tight_layout()
    plt.show()

def format_number(num):
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}K"  # Thousands
    elif num < 1000000000:
        return f"{num/1000000:.1f}M"  # Millions
    else:
        return f"{num/1000000000:.1f}B"  # Billions


def display_hist2(main_dict, unique_keys):
    # Create a list of keys from main_dict
    pic_keys = list(main_dict.keys())

    # Set the number of images per bar
    images_per_bar = 100

    # Set the number of bars to display
    num_bars = int(np.ceil(len(pic_keys) / images_per_bar))

    # Initialize a dictionary to store the cumulative counts
    cumulative_counts = {}

    # Loop over groups of images and calculate the cumulative counts
    for i in range(num_bars):
        # Get the keys for the current group of images
        start_idx = i * images_per_bar
        end_idx = min(start_idx + images_per_bar, len(pic_keys))
        group_keys = pic_keys[start_idx:end_idx]

        # Initialize a dictionary to store the counts for the current group
        group_counts = {}

        # Loop over the images in the current group and accumulate the counts
        for key in unique_keys:
            group_counts[key] = 0

        for key in group_keys:
            for slovo in unique_keys:
                dict_current = main_dict[key]
                if slovo in dict_current.keys():
                    if slovo not in group_counts:
                        group_counts[slovo] = 0
                    group_counts[slovo] += dict_current[slovo]

        # Add the counts for the current group to the cumulative counts
        for obj, count in group_counts.items():
            if obj not in cumulative_counts:
                cumulative_counts[obj] = []
            cumulative_counts[obj].append(count)

    # Sort the objects by their total count
    objects_sorted = sorted(cumulative_counts.keys(), key=lambda x: sum(cumulative_counts[x]), reverse=True)

    # Plot the histogram
    fig, ax = plt.subplots()

    bottom = np.zeros(num_bars)
    colors = ['#'+''.join(random.choices('0123456789ABCDEF', k=6)) for i in range(len(cumulative_counts.keys()))]

    for obj in objects_sorted:
        counts = cumulative_counts[obj]
        ax.bar(np.arange(num_bars), counts, bottom=bottom, color=colors.pop(), label=obj)
        bottom += counts

    # Add legend and labels
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    ax.set_xticks(list(range(0, 8500//images_per_bar)))
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_xlabel('Group of images ({} per group)'.format(images_per_bar))
    ax.set_ylabel('Count')
    ax.set_title('Cumulative histogram of objects in picture groups')

    plt.show()

def normalize_dicts(dicts):
    # Gather all keys from all dictionaries
    all_keys = set().union(*[d.keys() for d in dicts])
    # Ensure all dictionaries have the same keys
    for d in dicts:
        missing_keys = all_keys - d.keys()
        for key in missing_keys:
            d[key] = 0  # Set missing keys to zero
    return dicts

def sort_dicts_by_first(dict1, dict2, dict3):
    # Check if all dictionaries have the same keys
    if set(dict1.keys()) == set(dict2.keys()) == set(dict3.keys()):
        # Order dict2 and dict3 according to the key order of dict1
        ordered_dict2 = {key: dict2[key] for key in dict1.keys()}
        ordered_dict3 = {key: dict3[key] for key in dict1.keys()}
        return dict1, ordered_dict2, ordered_dict3
    else:
        # Return False if dictionaries do not have the same keys
        return dict1, dict2, dict3

def display_hist3(dict1,dict2,dict3):
    normalize_dicts([dict1,dict2,dict3])
    dict1,dict2,dict3 = sort_dicts_by_first(dict1,dict2,dict3)
    # Extract keys and values
    classes = list(dict1.keys())
    values1 = list(dict1.values())
    values2 = list(dict2.values())
    values3 = list(dict3.values())

    # Setting the positions and width for the bars
    positions = np.arange(len(classes))
    width = 0.25  # the width of a bar

    # Plotting
    plt.style.use('bmh')
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    bar1 = ax.bar(positions - width, values1, width, label='Full Railsem19 dataset')
    bar2 = ax.bar(positions, values2, width, label='Railsem19 test subset')
    bar3 = ax.bar(positions + width, values3, width, label='Railsem19 train subset')
    
    # Set labels, title, and tick parameters with specific font sizes
    ax.set_xlabel('Class Name', fontsize=24)  # Larger font size for x-axis label
    ax.set_ylabel('Classes Presence [%]', fontsize=24)   # Larger font size for y-axis label
    ax.set_xticks(positions)
    ax.set_xticklabels(classes, rotation=50, fontsize=16)  # Larger and rotated x-tick labels
    ax.set_yticklabels(ax.get_yticks(), fontsize=16)  # Larger y-tick labels
    ax.legend(fontsize=16)  # Larger legend font size

    # Show the plot
    plt.show()

if __name__ == "__main__":
    all_json = files_in_subdirs("RailNet_DT/rs19_val/jsons/", pattern = ["*.json"])
    mask_train = files_in_subdirs("RailNet_DT/rs19_val/uint8/rs19_val/", pattern = ["*.png"])
    masks_test = files_in_subdirs("RailNet_DT/rs19_val/uint8/test/", pattern = ["*.png"])
    
    global_pixel_count = {}
    
    objects_exported_test, global_pixel_count = export_objects_mask(masks_test, translate_dict, global_pixel_count)
    objects_exported_train, global_pixel_count = export_objects_mask(mask_train, translate_dict, global_pixel_count)
    
    sorted_dict = {k: global_pixel_count[k] for k in sorted(global_pixel_count)}
    display_hist1(sorted_dict)
    
    obj_copy = objects_exported_train.copy()
    objects_exported_testt = {key + 7649: value for key, value in objects_exported_test.items()}
    obj_copy.update(objects_exported_testt)
    objects_exported_all = obj_copy
    
    #objects_exported = export_objects_json(all_json)
    
    all_keys_count,file_all_keys_count,file_unique_keys_count,unique_keys,img_count,key_not_present_count = get_stats(objects_exported_all)
    train_keys_count,file_train_keys_count,train_unique_keys_count,unique_keys_train,img_count_train,key_not_present_count_train = get_stats(objects_exported_train)
    test_keys_count,file_test_keys_count,test_unique_keys_count,unique_keys_test,img_count_test,key_not_present_count_test = get_stats(objects_exported_test)
    display_hist1(key_not_present_count)
    display_hist1(all_keys_count)
    display_hist2(objects_exported_all, unique_keys)
    
    dict_rel_all = {key: (value / len(all_json))*100 for key, value in all_keys_count.items()}
    dics_rel_test = {key: (value / len(masks_test))*100 for key, value in test_keys_count.items()}
    dict_rel_train = {key: (value / len(mask_train))*100 for key, value in train_keys_count.items()}

    display_hist3(dict_rel_all,dics_rel_test,dict_rel_train)

    mean_general = np.array(list(file_all_keys_count.values())).mean()
    mean_unique = np.array(list(file_unique_keys_count.values())).mean()
    max_unique = np.array(list(file_unique_keys_count.values())).max()
    max_general = np.array(list(file_all_keys_count.values())).max()
    max_general_idx = list(file_all_keys_count.values()).index(max_general)
    max_unique_idx =  list(file_all_keys_count.values()).index(max_unique)
    print(max_unique)
    print(max_unique_idx)
    most_obj = objects_exported_all[990]
    print(most_obj)
    min_unique = np.array(list(file_unique_keys_count.values())).min()
    min_general = np.array(list(file_all_keys_count.values())).min()

    print('done')