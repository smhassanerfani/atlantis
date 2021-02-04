# Chose Images and correspond json_file is located in the same directory
# throgh code we get access to the directory and extract the name of each image file (id)
import os
import json

rootdir = "/home/serfani/Downloads/new_dataset"

labels_list = list()
with os.scandir(rootdir) as it:
    for entry in it:
        if entry.is_dir():
            labels_list.append(entry.name)

creativecommons = [1, 2, 3, 4, 5, 6, 7, 8]

images_list = list()

for label in labels_list:
    json_list = []
    for license in creativecommons:
        label_path = os.path.join(rootdir, label)
        license_path = os.path.join(label_path, str(license))

        with os.scandir(license_path) as it:
            for entry in it:
                if entry.name.endswith('.jpg'):
                    images_list.append(entry.name.split('.')[0])
        try:
            with open(os.path.join(license_path, 'json_file.json'), ) as jf:
                img_json = json.load(jf)
        except FileNotFoundError as error:
            print(f'FileNotFoundError: {error}')

        for imgID in images_list:

            dic_to_json = {}
            for img in img_json['photos']['photo']:
                if imgID == img['id']:
                    try:
                        dic_to_json['id'] = img['id']
                        # dic_to_json["secret"] = img["secret"]
                        # dic_to_json["server"] = img["server"]
                        # dic_to_json["farm"] = img["farm"]
                        dic_to_json["license"] = img["license"]
                        dic_to_json["flickr_url"] = img["flickr_url"]
                        # dic_to_json["file_name"] = img["file_name"]
                        dic_to_json["date_captured"] = img["date_captured"]
                        dic_to_json["width"] = img["width"]
                        dic_to_json["height"] = img["height"]

                    except KeyError as error:
                        print(f'KeyError: {error}')

                    json_list.append(dic_to_json)
                    break

    with open(os.path.join(label_path, 'json_file.json'), 'a') as jf2:
        json.dump(json_list, jf2, indent=4)
