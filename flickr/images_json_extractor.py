import os
import json
import argparse


IMAGES_ROOT = "/home/serfani/Downloads/json_project/images"
LABEL_DIR = "/home/serfani/Downloads/json_project/"
JSON_DIR = "/home/serfani/Downloads/json_project/"


def get_arguments():
    parser = argparse.ArgumentParser(
        description="This code parses and captures the corresponding information of final images from initial json file and saves in new json file.")
    parser.add_argument("-i", "--images-root", type=str, default=IMAGES_ROOT, required=False,
                        help="The path of final images.")
    parser.add_argument("-l", "--label-dir", type=str, default=LABEL_DIR, required=False,
                        help="The path of directory including the first images based on different CC license code.")
    parser.add_argument("-j", "--json-dir", type=str, default=JSON_DIR, required=False,
                        help="The path the reparsed json will be dumped.")

    return parser.parse_args()


args = get_arguments()


def get_images_list(image_root):
    image_root = os.path.join(image_root, "images")
    images_list = list()
    for root, dirs, files in os.walk(image_root, topdown=True):
        for file in files:
            if file.endswith(".jpg"):
                images_list.append(file.split(".")[0])

    return images_list


def get_image_info(label_dir, images_list):
    json_list = list()
    for root, dirs, files in os.walk(label_dir, topdown=True):
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
        args.label_dir, get_images_list(args.images_root))
    dump_json_info(args.json_dir, json_list)
    print(f"number of images were parsed: {len(json_list)}")


if __name__ == '__main__':
    main()
