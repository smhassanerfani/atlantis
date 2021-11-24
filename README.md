# ATLANTIS - ArTificiaL And Natural waTer-bodIes dataSet
This is the respository for the ATLANTIS. All waterbody labels are comprehensively described in [ATLANTIS Wiki](https://github.com/smhassanerfani/atlantis/wiki). This dataset developed in the [iWERS](http://ce.sc.edu/goharian/) lab at the University of South Carolina.
<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atlantis/blob/master/wiki/dataset.png">
  Figure 1. ATLANTIS - ArTificiaL And Natural waTer-bodIes dataSet.
</p>

## Overview
ATLANTIS is a benchmark for semantic segmentation of waterbody images. For the first time, this dataset covers a wide range of natural waterbodies such as sea, lake, river and man-made (artificial) water-related sturcures such as dam, reservoir, canal, and pier. ATLANTIS includes 5,195 pixel-wise annotated images split to 3,364 training, 535 validation, and 1,296 testing images. In addition to 35 waterbodies, this dataset covers 21 general labels such as person, car, road and building.

## AQUANet
In addition to waterbodies dataset, and in order to tackle the inherent challenges in the segmentation of waterbodies, we ([iWERS](http://ce.sc.edu/goharian/) in collaboration with [Computer Vision Lab](https://cvl.cse.sc.edu/)) developed a CNN-based semantic segmentation network which takes advantage of two different paths to process the aquatic and non-aquatic regions, separately. Each path includes low-level feature and cross-path modulation, to adjust features for better representation. The results show that AQUANet outperforms other state-of-the-art semantic-segmentation networks on ATLANTIS, and the ablation studies justify the effectiveness of the proposed components.
<p align="center">
  <img width="120%" height="120%" src="https://github.com/smhassanerfani/atlantis/blob/master/wiki/aquanet.svg">
  Figure 2. The network architecture of proposed AQUANet.
</p>


## Dataset Description
The ATLANTIS dataset is designed and developed with the goal of capturing a wide-range of water-related objects, either those exist in natural environment or the infrastructure and man-made (artificial) water systems. In this dataset, labels were first selected based on the most frequent objects, used in water-related studies or can be found in real-world scenes. Aside from the background objects, total of 56 labels, including 17 artificial, 18 natural water-bodies, and 21 general labels, are selected. These general labels are considered for providing contextual information that most likely can be found in water-related scenes. After finalizing the selection of waterbody labels, a comprehensive investigation on each individual label was performed by annotators to make sure all the labels are vivid examples of those objects in real-world. Moreover, sometimes some of the water-related labels, e.g., levee, embankment, and floodbank, have been used interchangeably in water resources field; thus, those labels are either merged into a unique group or are removed from the dataset to prevent an individual object receives different labels.

In order to gather a corpus of images, we have used Flickr API to query and collect 800 "medium-sized" unique images for each label based on seven commonly used "Creative Commons" and "United States Government Work" licenses. Downloaded images were then filtered by a two-stage hierarchical procedure. In the first stage, each annotator was assigned to review a specific list of labels and remove irrelevant images based on that specific list of labels. In the second stage, several meetings were held between the entire annotation team and the project coordinator to finalize the images which appropriately represent each of 56 labels. Finally, images were annotated by annotators who have solid water resources engineering background as well as experience working with the [CVAT](https://github.com/openvinotoolkit/cvat), which is a free, open source, and web-based image/video annotation tool. If you wish to contribute to this project or you want to develop a semantic segmentation dataset, please check [ATLANTIS DEVELOPMENT KIT](https://github.com/smhassanerfani/atlantis/tree/master/adk). If you are curious how images are annotated in this project, please watch the follwoing tutorial videos:

<TABLE>
  <TR>
     <TD><a href="https://youtu.be/HD9_MBwlGFE"><img src="https://github.com/smhassanerfani/atlantis/blob/master/wiki/46882608915.png" width="100%" /></a></TD>
     <TD><a href="https://youtu.be/5JvMjWNVkKM"><img src="https://github.com/smhassanerfani/atlantis/blob/master/wiki/49216008106.png" width="100%" /></a></TD>
     <TD><a href="https://youtu.be/wsnWqU6EnGo"><img src="https://github.com/smhassanerfani/atlantis/blob/master/wiki/9954579843.png" width="100%" /></a></TD>
  </TR>
  <TR>
     <TD align="center"><a href="https://youtu.be/HD9_MBwlGFE">(Watch Video 1)</a></TD>
     <TD align="center"><a href="https://youtu.be/5JvMjWNVkKM">(Watch Video 2)</a></TD>
     <TD align="center"><a href="https://youtu.be/wsnWqU6EnGo">(Watch Video 3)</a></TD>
  </TR>
</TABLE>

## Dataset Statistics

Figure 3 shows the frequency distribution of the number of images for waterbody labels. Labels are ranked based on pixel frequencies.
<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atlantis/blob/master/wiki/frequency_distribution.svg">
  Figure 3. Frequency distribution of the number of images assigned to each waterbody label.
</p>

Figure 4 shows the frequency distribution of the number of pixels for all 56 ATLANTIS labels plus background (percentage).
<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atlantis/blob/master/wiki/pixels_frequency_distribution.svg">
  Figure 4. Percentage of pixels frequency distribution for all 56 ATLANTIS labels plus background.
</p>

Such a long-tailed distribution is common for semantic segmentation datasets even if the number of images that contain specific label are pre-controlled. Such frequency distribution for pixels would be inevitable for objects existing in real-world. Taking "water tower" as an example, despite having 219 images, the number of pixels are less than many other labels in the dataset. In total, only 4.89% of pixels are unlabeled, and 34.17% and 60.94% of pixels belong to waterbodies (natural and artificial) and general labels, respectively. 
<p align="center">
  <img width="100%" height="100%" src="https://github.com/smhassanerfani/atlantis/blob/master/wiki/r2_images_vs_pixels.svg">
  Figure 5.  The "R-squared" of the regression between the number of images and pixels is relatively low.
</p>

# Reference

```
@misc{erfani2021atlantis,
      title={ATLANTIS: A Benchmark for Semantic Segmentation of Waterbody Images}, 
      author={Seyed Mohammad Hassan Erfani and Zhenyao Wu and Xinyi Wu and Song Wang and Erfan Goharian},
      year={2021},
      eprint={2111.11567},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
