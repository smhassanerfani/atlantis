# ATLANTIS - ArTificiaL And Natural waTer-bodIes dataSet
## A Benchmark for Semantic Segmentation of Waterbody Images
![](https://github.com/smhassanerfani/atlantis/blob/master/wiki/dataset.png)
This is the respository for the ATLANTIS. All waterbody labels are comprehensively described in [ATLANTIS Wiki](https://github.com/smhassanerfani/atlantis/wiki).

For the first time, this dataset covers a wide range of natural waterbodies such as sea, lake, river and man-made (artificial) water-related sturcures such as dam, reservoir, canal, and pier. ATLANTIS includes 5,195 pixel-wise annotated images split to 3,364 training, 535 validation, and 1,296 testing images. In addition to 35 waterbodies, this dataset covers 21 general labels such as person, car, road and building.

## AQUANet
In addition to waterbodies dataset, and in order to tackle the inherent challenges in the segmentation of waterbodies, we also developed, CNN-based semantic segmentation network, which takes advantage of two different paths to process the aquatic and non-aquatic regions, separately. Each path includes low-level feature and cross-path modulation, to adjust features for better representation. The results show that AQUANet outperforms other state-of-the-art semantic-segmentation networks on ATLANTIS, and the ablation studies justify the effectiveness of the proposed components.
