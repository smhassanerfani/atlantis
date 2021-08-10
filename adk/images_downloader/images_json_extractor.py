import os
import json
import argparse


IMAGES_DIR = "/home/serfani/Downloads/atlantis/s1a/breakwater/images"
LICENSE_DIRS = "/home/serfani/Downloads/atlantis/s1a/breakwater/"
JSON_DIR = "/home/serfani/Downloads/atlantis/s1a/breakwater/"


def get_arguments():
    parser = argparse.ArgumentParser(
        description="This code parses and captures the corresponding information of final images from initial json file and saves in new json file.")
    parser.add_argument("-i", "--images-dir", type=str, default=IMAGES_DIR, required=False,
                        help="The path of final annotated images.")
    parser.add_argument("-l", "--license-dirs", type=str, default=LICENSE_DIRS, required=False,
                        help="The path of 8 license directories including the primary requested images.")
    parser.add_argument("-j", "--json-dir", type=str, default=JSON_DIR, required=False,
                        help="The path that the reparsed json will be dumped.")

    return parser.parse_args()


args = get_arguments()


def get_images_list(images_dir):
    images_dir = os.path.join(images_dir, "images")
    images_list = list()
    for root, dirs, files in os.walk(images_dir, topdown=True):
        for file in files:
            if file.endswith(".jpg"):
                images_list.append(file.split(".")[0])

    return images_list


def get_image_info(license_dirs, images_list):
    json_list = list()
    for root, dirs, files in os.walk(license_dirs, topdown=True):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), ) as jf:
                    json_info = json.load(jf)

                for image_name in images_list:
                    for entry in json_info['photos']['photo']:
                        if image_name == entry['id']:
                            # print(image_name)
                            try:
                                json_list.append({
                                    "id": entry["id"],
                                    # "owner": entry["owner"],
                                    "secret": entry["secret"],
                                    "server": entry["server"],
                                    "farm": entry["farm"],
                                    # "title": entry["title"],
                                    # "ispublic": entry["ispublic"],
                                    "license": entry["license"],
                                    "flickr_url": entry["flickr_url"],
                                    "file_name": entry["file_name"],
                                    "date_captured": entry["date_captured"],
                                    "width": entry["width"],
                                    "height": entry["height"]
                                })

                            except KeyError as err:
                                print(f'KeyError: {err}')
                            break

    return json_list


def dump_json_info(json_root, json_list):
    with open(os.path.join(json_root, 'json_file.json'), 'a') as jf:
        json.dump(json_list, jf, indent=4)


def main():

    json_list = get_image_info(
        args.license_dirs, get_images_list(args.images_dir))
    dump_json_info(args.json_dir, json_list)
    print(f"number of images were parsed: {len(json_list)}")


if __name__ == '__main__':
    main()
