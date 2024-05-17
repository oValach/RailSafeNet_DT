import json
import os

jsons_path = "RailNet_DT\\rs19_val\\jsons\\rs19_val"

objects = {}
for filename in os.listdir(jsons_path):
    data = json.load(open(os.path.join(jsons_path,filename), 'r'))
    data_objects = data["objects"]
    for object in data_objects:
        if object["label"]:
            if object["label"] not in objects.keys():
                objects[object["label"]] = 1
            else:
                objects[object["label"]] += 1
            
print(objects)