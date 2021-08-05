# ADK - ATLANTIS DEVELOPMENT KIT

In order to expand the size of ATLANTIS and minimize the effort to address challenges associated with dataset development, we initiate ATLANTIS DEVELOPMENT KIT as an open community effort where experts in water resources communities can contribute in annotating and adding images to ATLANTIS. We developed different pipelines to facilitate downloading, annotating, organizing and analyzing new images. Here, we explained how you can use these pipelines whether for contribution to this project or for personal use.

## What's included

Within the download you'll find the following directories and files. You'll see something like this:

```text
adk/
    ├── dataset
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
    │   ├── pipeline.sh
    │   └── spatial_analysis.py
    ├── images_downloader
    │   ├── images_downloader.py
    │   └── images_json_extractor.py
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
