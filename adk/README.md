# ADK - ATLANTIS DEVELOPMENT KIT

In order to expand the size of ATLANTIS and minimize the effort to address challenges associated with dataset development, we initiate ATLANTIS DEVELOPMENT KIT as an open community effort where experts in water resources communities could contribute in annotating and adding images to ATLANTIS. We develop different pipelines to facilitate downloading, annotating, organizing and analyzing new images. Here, we will explain how you can use these pipelines whether for contribution to this project or for personal use.

## What's included

Within the download you'll find the following directories and files. You'll see something like this:

```text
adk/
├── dataset/
│   ├── sa1/
|   .   ├── breakwater/
|   .   |   ├── images/
|   .   |   ├── masks/
|       |   ├── rgb_masks/
|       |   ├── annotations.xml
|       |   └── flickr.json
|       └── bridge/
|       .   ├── images/
|       .   .
|       .   .
|           .
|   .
|   .
|   .
|   └── s4
|       ├── dam/
|       .
|       .
|       .
|
├── images_analyser/
|   ├── color2id.py
|   ├── pipeline.sh
|   └── spatial_analysis.py   
├── images_downloader/
├── images_organizer/
├── inconsistency_analysis/
└── waterbody_extractor   
```
