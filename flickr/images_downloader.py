import os
import csv
import json
import requests
from datetime import datetime
from PIL import Image
import io

# Key:
api_key = 'da03d4d2c9d70753c5b918009711b9ea'

# Secret:
secret = 'f35c6f99121e3e83'

# natural_list = list()
# artificial_list = list()
# path = "./"
# with open(path, encoding="utf8", errors='ignore') as csvfile:
#     csv_list = csv.DictReader(csvfile)
#     for row in csv_list:
#         natural_list.append(row['natural'])
#         artificial_list.append(row['artificial'])

# natural_list = list(filter(None, natural_list))
# artificial_list = list(filter(None, artificial_list))

labels_list = ["lake"]

licenses = [
    {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"},
    {"url": "http://creativecommons.org/licenses/by-nc/2.0/",
        "id": 2, "name": "Attribution-NonCommercial License"},
    {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
        "id": 3, "nameswimming pool": "Attribution-NonCommercial-NoDerivs License"},
    {"url": "http://creativecommons.org/licenses/by/2.0/",
        "id": 4, "name": "Attribution License"},
    {"url": "http://creativecommons.org/licenses/by-sa/2.0/",
        "id": 5, "name": "Attribution-ShareAlike License"},
    {"url": "http://creativecommons.org/licenses/by-nd/2.0/",
        "id": 6, "name": "Attribution-NoDerivs License"},
    {"url": "http://flickr.com/commons/usage/", "id": 7,
        "name": "No known copyright restrictions"},
    {"url": "http://www.usa.gov/copyright.shtml",
        "id": 8, "name": "United States Government Work"}
]

# os.chdir('/home/serfani/Downloads/new_dataset')
# root_path = os.getcwd()
root_path = '/home/serfani/Downloads/new_dataset'

try:
    os.makedirs(root_path)
except FileExistsError:
    pass

for label in labels_list:
    try:
        label_path = os.path.join(root_path, label)
        os.makedirs(label_path)
    except FileExistsError:
        pass

    for license in licenses:
        license_path = os.path.join(label_path, str(license["id"]))
        try:
            os.makedirs(license_path)
        except FileExistsError:
            pass

        tag = label
        license_type = license["id"]
        per_page = 50
        page_number = 1

        url = 'https://www.flickr.com/services/rest/?method=flickr.photos.search&api_key={api_key}&tags={tag}&tag_mode={tm}&license={license_type}&min_upload_date={date}&per_page={per_page}&page={page_number}&format=json&nojsoncallback=1'.format(
            api_key=api_key, tag=tag, tm="all", license_type=license_type, date=1577836801, per_page=per_page, page_number=page_number)

        responses = requests.get(url)
        data = responses.json()

        for element in data['photos']['photo']:
            # element['title'], element['owner'], element['ispublic']
            del element['isfriend'], element['isfamily']
            element['license'] = license["id"]
            element['flickr_url'] = 'https://farm{farm_id}.staticflickr.com/{server_id}/{id}_{secret}_{size}.jpg'.format(
                farm_id=element['farm'], server_id=element['server'], id=element['id'], secret=element['secret'], size="z")
            element['file_name'] = str(element['id']) + '.jpg'
            element['date_captured'] = datetime.now(
            ).strftime("%m/%d/%Y, %H:%M:%S")

            # add width and length information of the images
            try:
                response = requests.get(element['flickr_url'], timeout=6.0)
                image_bytes = io.BytesIO(response.content)
                img = Image.open(image_bytes)
                element['width'], element['height'] = img.size

                image_path = os.path.join(license_path, element['file_name'])
                print(image_path)

                with open(image_path, "wb") as handler1:
                    handler1.write(response.content)

            except OSError as error:
                print(f"OSError: {str(error)}")
            except requests.ConnectionError as error:
                print(f"ConnectionError: {str(error)}")

        # dump a json file
        with open(os.path.join(license_path, 'json_file.json'), 'w') as tf:
            json.dump(data, tf, indent=4)
            print(f"{license_path} is done!")

print("it's finish")
