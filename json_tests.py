import json
import os

jsons_path = "RailNet_DT\\rs19_val\\jsons\\rs19_val"

person_files = 0
persong_files = 0
for filename in os.listdir(jsons_path):
    data = json.load(open(os.path.join(jsons_path,filename), 'r'))
    data_objects = data["objects"]
    for object in data_objects:
        if object["label"] == "person":
            person_files += 1
        if object["label"] == "person-group":
            persong_files += 1

print(person_files)
print(persong_files)