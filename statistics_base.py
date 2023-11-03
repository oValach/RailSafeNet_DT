import os, glob
import numpy as np
import json
import random
from ProgressBar import printProgressBar
import matplotlib.pyplot as plt

def files_in_subdirs(start_dir, pattern = ["*.png","*.jpg","*.jpeg"]):
    files = []
    for p in pattern:
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob.glob(os.path.join(dir,p)))
    return files

def export_objects(json_files):
    f_index = 0
    objects = {}
    files = len(json_files)
    #printProgressBar(0, files, prefix = 'Progress:', suffix = 'Complete', length = 50)
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
            #printProgressBar(f_index + 1, files, prefix = 'Loading statistics:', suffix = 'Complete', length = 50)

    return objects

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
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    # Plot the histogram
    rects = ax.bar(keys, values, color='#8B0000')

    # Add text to each bar
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height+200, str(values[i]), ha='center', va='center', color='black')

    # Set the labels and title
    ax.set_xlabel('Items')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of objects not present at pictures')

    # Rotate the x-axis labels by 90 degrees
    plt.xticks(rotation=25)

    # Show the plot
    plt.show()

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

if __name__ == "__main__":
    all_json = files_in_subdirs("./rs19_val/jsons/", pattern = ["*.json"])
    
    objects_exported = export_objects(all_json)
    all_keys_count,file_all_keys_count,file_unique_keys_count,unique_keys,img_count,key_not_present_count = get_stats(objects_exported)
    display_hist1(key_not_present_count)
    display_hist1(all_keys_count)
    display_hist2(objects_exported, unique_keys)

    mean_general = np.array(list(file_all_keys_count.values())).mean()
    mean_unique = np.array(list(file_unique_keys_count.values())).mean()
    max_unique = np.array(list(file_unique_keys_count.values())).max()
    max_general = np.array(list(file_all_keys_count.values())).max()
    max_general_idx = list(file_all_keys_count.values()).index(max_general)
    max_unique_idx =  list(file_all_keys_count.values()).index(max_unique)
    print(max_unique)
    print(max_unique_idx)
    most_obj = objects_exported[990]
    print(most_obj)
    min_unique = np.array(list(file_unique_keys_count.values())).min()
    min_general = np.array(list(file_all_keys_count.values())).min()

    print('done')