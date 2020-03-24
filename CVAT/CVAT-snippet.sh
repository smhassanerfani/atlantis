sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

sudo groupadd docker
sudo usermod -aG docker $USER

sudo apt-get install -y python3-pip
sudo python3 -m pip install docker-compose

sudo apt-get install -y git
git clone https://github.com/opencv/cvat
cd cvat

docker-compose build
docker-compose up -d

docker exec -it cvat bash -ic 'python3 ~/manage.py createsuperuser'

curl https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
sudo apt-get update
sudo apt-get install -y google-chrome-stable

# Build docker image
docker-compose -f docker-compose.yml -f components/openvino/docker-compose.openvino.yml build
docker-compose -f docker-compose.yml -f components/openvino/docker-compose.openvino.yml up -d

# Semi-Automatic Segmentation with Deep Extreme Cut:
# using deep learning models for semi-automatic semantic segmentation. get a segmentation polygon from four (or more) extreme points of an object.
# based on the pre-trained DEXTR model which has been converted to Inference Engine format.
docker-compose -f docker-compose.yml -f components/openvino/docker-compose.openvino.yml -f cvat/apps/dextr_segmentation/docker-compose.dextr.yml build
docker-compose -f docker-compose.yml -f components/openvino/docker-compose.openvino.yml -f cvat/apps/dextr_segmentation/docker-compose.dextr.yml up -d

# Kibana
docker-compose -f docker-compose.yml -f components/analytics/docker-compose.analytics.yml build
docker-compose -f docker-compose.yml -f components/analytics/docker-compose.analytics.yml up -d

# Fast R-CNN
docker-compose -f docker-compose.yml -f components/tf_annotation/docker-compose.tf_annotation.yml build
docker-compose -f docker-compose.yml -f components/tf_annotation/docker-compose.tf_annotation.yml up -d

# Keras+Tensorflow Mask R-CNN Segmentation:
# automatical segment many various objects on images, pre-trained model on MS COCO dataset, based on Feature Pyramid Network (FPN) and a ResNet101 backbone.
docker-compose -f docker-compose.yml -f components/auto_segmentation/docker-compose.auto_segmentation.yml build
docker-compose -f docker-compose.yml -f components/auto_segmentation/docker-compose.auto_segmentation.yml up -d

docker-compose -f docker-compose.yml -f components/openvino/docker-compose.openvino.yml -f cvat/apps/dextr_segmentation/docker-compose.dextr.yml -f components/auto_segmentation/docker-compose.auto_segmentation.yml build
docker-compose -f docker-compose.yml -f components/openvino/docker-compose.openvino.yml -f cvat/apps/dextr_segmentation/docker-compose.dextr.yml -f components/auto_segmentation/docker-compose.auto_segmentation.yml up -d