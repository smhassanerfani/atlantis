import os
import csv
import json
import requests
from datetime import datetime
from PIL import Image
import io
import argparse

DATASET_ROOT = "./"
LABELS_LIST = ["lake", "river"]
IMAGES_NUMBER = 5


def get_arguments():
    parser = argparse.ArgumentParser(
        description="This code dowloads images and images attributes from Flickr databes.")
    parser.add_argument("-k", "--api-key", type=str,
                        required=True, help="Flickr API Key.")
    parser.add_argument("-n", "--images-number", type=int, default=IMAGES_NUMBER,
                        required=False, help="Number of requests (images) for each license.")
    parser.add_argument("-r", "--dataset-root", type=str, default=DATASET_ROOT, required=False,
                        help="The path label directories are stored.")
    parser.add_argument("-l", "--labels-list", type=str, nargs="+", default=LABELS_LIST, required=False,
                        help="The list of labels.")

    return parser.parse_args()


args = get_arguments()


def makedir(dirname):
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass


with open("licenses_info.json", 'r') as jf:
    licenses = json.load(jf)

makedir(args.dataset_root)

for label in args.labels_list:

    label_path = os.path.join(args.dataset_root, label)
    makedir(label_path)

    for license in licenses:

        license_path = os.path.join(label_path, str(license["id"]))
        makedir(license_path)

        tag = label
        license_type = license["id"]
        per_page = args.images_number
        page_number = 1

        # https://www.flickr.com/services/api/flickr.photos.search.html
        url = 'https://www.flickr.com/services/rest/?method=flickr.photos.search&api_key={api_key}&tags={tag}&tag_mode={tm}&license={license_type}&per_page={per_page}&page={page_number}&format=json&nojsoncallback=1'.format(
            api_key=args.api_key, tag=tag, tm="all", license_type=license_type, per_page=per_page, page_number=page_number)

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

    print(f"Downling \"{label}\" images is finish!")
