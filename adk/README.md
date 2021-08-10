# ADK - ATLANTIS DEVELOPMENT KIT

In order to expand the size of ATLANTIS and minimize the effort to address challenges associated with dataset development, we initiate ATLANTIS DEVELOPMENT KIT as an open community effort where experts in water resources communities can contribute in annotating and adding images to ATLANTIS. We developed different pipelines to facilitate downloading, annotating, organizing and analyzing new images. Here, we explained how you can use these pipelines whether for contribution to this project or for personal use.

## Table of contents

- [What's included](#whats-included)
- [`dataset/`](#dataset)
- [`images_analyser/`](#images_analyser)
- [`images_downloader/`](#images_downloader)

## What's included

Within the download you'll find the following directories and files. You'll see something like this:

```text
adk/
    ├── dataset
    │   ├── cvat_labels_constructor.json 
    │   ├── image_directories.txt
    │   ├── images_list.txt
    │   ├── s1a
    │   │   ├── breakwater
    │   │   │   ├── annotations.xml
    │   │   │   ├── flickr.json
    │   │   │   ├── images
    │   │   │   ├── masks
    │   │   │   └── rgb_masks
    │   │   ├── bridge
    │   │   ├── canal
    │   │   ├── culvert
    │   │   .
    │   │   .
    │   │   .
    │   │   └── water_well
    │   ├── s1n
    │   │   ├── cliff
    │   │   ├── cypress_tree
    │   │   ├── fjord
    │   │   ├── flood
    │   │   .
    │   │   .
    │   │   .
    │   │   └── wetland
    │   ├── s2
    │   │   ├── canal
    │   │   ├── canal_amsterdam
    │   │   ├── cypress_tree
    │   │   .
    │   │   .
    │   │   .
    │   │   └── wetland
    │   ├── s3
    │   │   ├── bridge
    │   │   ├── bridge_brooklyn_bridge
    │   │   ├── bridge_drawbridge
    │   │   .
    │   │   .
    │   │   .
    │   │   └── swimming_pool_pool
    │   ├── s4
    │   │   ├── dam
    │   │   ├── fjord
    │   │   ├── flood
    │   │   .
    │   │   .
    │   │   .
    │   │   └── wetland_bog
    │   └── xml_extractor.py
    ├── images_analyser
    │   ├── color2id.py
    │   ├── labels_info.json
    │   └── pipeline.sh
    ├── images_downloader
    │   ├── images_downloader.py
    │   └── images_json_extractor.py
    │   └── licenses_info.json
    │   └── pipeline.sh
    ├── images_organizer
    │   ├── artificial_dataloader.sh
    │   ├── cp_code_train_val_test.sh
    │   ├── images_masks_diff_finder.sh
    │   ├── images_pooling.sh
    │   └── train_val_test_distributor.sh
    ├── inconsistency_analysis
    │   ├── first_edition
    │   │   ├── ammar
    │   │   ├── ashlin
    │   │   ├── ground_truth
    │   │   ├── reddy
    │   │   └── tripp
    │   ├── models_evaluation
    │   │   ├── compute_iou.py
    │   │   ├── labels_ID.json
    │   │   ├── pipeline.sh
    │   │   └── requirements.txt
    │   └── second_edition
    │       ├── ammar
    │       ├── ashlin
    │       └── tripp
    └── waterbody_extractor
        ├── labels_ID.json
        ├── pipeline.sh
        └── pyscript.py
```

## `dataset/`
The first directory in ADK is dataset. This directory is comprised of one json file, two text files, a pysciprt and five sub-directories representing different serieses which images are downloaded, annotated, analyzed and oranized.

### `./cvat_labels_constructor.json`
This json file includes all ATLANTIS labels compatible with [CVAT](https://github.com/openvinotoolkit/cvat). To add all 56 labels to CVAT, users can use this code.

### `./image_directories.txt`
This text file lists image addresses in all five sub-directories.

### `./image_list.txt`
This text file inludes the name list of images which exists in this dataset. This id name is unique and created by Flickr.

### `./s1a/breakwater/`
Each sub-directories includes number of sub-sub-directories named with respect to one of waterbody labels. Considering `dataset/s1a/breakwater/` as an example, it consists of following files and directories:

* `annotations.xml`: Annotation file exproted from CVAT. Images located in `images/` along with `annotations.xml` enables further modification on annotations by users. For this purpose, users can install CVAT and create a task using images of `images/`. NOTE: the annotation labels must be constructed at this stage. Users can easily copy the label codes from `cvat_labels_constructor.json`. 

* `flickr.json`: In this file, all information associated with images of `images/` directory in [Flikr](https://www.flickr.com/) database are stored as follow:

```
[
    {
        "id": "30326712458",
        "secret": "a9ff6805e0",
        "server": "1899",
        "farm": 2,
        "license": 1,
        "flickr_url": "https://farm2.staticflickr.com/1899/30326712458_a9ff6805e0_z.jpg",
        "file_name": "30326712458.jpg",
        "date_captured": "01/13/2020, 23:25:19",
        "width": 640,
        "height": 427
    },
    .
    .
    .
```

* `images/`: This directory contains all `.jpg` images downloaded from Flickr.  
* `masks/`: This directory includes all `.png` id masks correspondence with `.jpg` in `images/`.   
* `rgb_masks/`: This directory includes all `.png` RGB masks correspondence with `.jpg` in `images/`.     

### `./xml_extractor.py`
Images chosen for annotation are usually determined through procedural rules. They must both be relevant to the label and representative. However, sometimes, annotators decide to skip some images during annotation. In this situation, the exported xml file records the information of all images in order (annotated or skipped). So, if annotators want to reload the project for further modification, they must still include the skipped images in the project. This `xml_extractor.py` helps users and annotators by removing any skipped images from the xml file, allowing the annotator to reload the project with only the chosen images.

## `images_analyser/`
This directory includes all python scripts that users need to convert rgb masks to id masks. First, labels information should be modified based on your dataset. `name` and `id` are considered based on the labels existing in the dataset, and `color` is the RGB code chosen for each label in CVAT. `color2id.py` creates both id masks and csv file which lists the number of pixels and segments of each label in the annotated masks existing in the directory. This information is useful for statistical analysis of number of pixels and segments in the dataset. Users can use the `pipeline.sh` to run the code through all image directories at once.

## `images_downloader/`
In order to download images from Flickr, users first need to apply for [Flickr API Key](https://www.flickr.com/services/api/misc.api_keys.html). Using `pipline.sh` data_root, api_key, labels_list and images_numbers should be set. NOTE: This code need [reuquests](https://docs.python-requests.org/en/master/) package, so before run the code, requests package should be installed.
Many Flickr users have chosen to offer their work under a [Creative Commons](https://www.flickr.com/creativecommons/) license. In addition to Creative Commons licenses, "No known copyright restrictions" and "United States Government Work" will be requested by this code. Images for each license will be stored in indiviudal directory. `licenses_info.json` lists the name, id and url address for all licenses.

After image download request, in each license directory, in addition to images, a `json_file.json` exists. This file lists the information attributed to downloaded images. It is so common that uers want to remove some irrelevant images. In order to update the `json_file.json` according to chosen images, `images_json_extractor.py` is provided. Using this code, users can easily eliminate the redandant records belonging to removed images.   