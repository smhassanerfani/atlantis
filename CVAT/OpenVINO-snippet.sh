# How to install OpenVINO
# Download the openvino_toolkit.tgz
# Go to the directory
cat /proc/cpuinfo | grep "model_name"
tar -xzf openvino_toolkit.tgz
cd openvino_directory
sudo -E ./install_openvino_dependencies.sh
sudo -E ./install_GUI.sh


# initialization
cd /opt/intel/openvino
source bin/setupvars.sh


# Inference Engines
# version1
cd /opt/intel/openvino/deployment_tools/open_model_zoo/demos
mkdir build
cd build
cmake ..
make
# version2
cd ~/Documents
mkdir OpenVINO-Samples-Build
cd OpenVINO-Samples-Build/
cmake /opt/intel/openvino/deployment_tools/open_model_zoo/demos
make

cd ~/Documents/OpenVINO-Samples-Build/
cd ~/Documents/OpenVINO-Samples-Build/intel64/Release/
# -h throgh this option you can check the important input and output parameters
./security_barrier_camera_demo -h
./security_barrier_camera_demo -i /opt/intel/openvino/deployment_tools/demo/car_1.bmp -m /opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.xml

# In case running to a problem
# sudo updatedb
# locate mydata
sudo cmake -D OpenCV_DIR="/opt/intel/openvino_2020.1.023/opencv/cmake/" -D InferenceEngine_DIR="/opt/intel/openvino/deployment_tools/inference_engine/share/" ..
sudo cmake -D InferenceEngine_DIR="/opt/intel/openvino/deployment_tools/inference_engine/share/" ..


# complete address of intel & public models in openvino directory
# models are here:
export models=/opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel
export intel_model=/opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/intel/
export public_model=/opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/public/

# look into the frozen .pb file (look at the instrucrure of algorithm)
export tf_pb_to_tb=/usr/local/lib/python3.6/dist-packages/tensorflow/python/tools/import_pb_to_tensorboard.py
export tf_gnv3=/opt/intel/openvino_2019.3.376/deployment_tools/open_model_zoo/tools/downloader/public/googlenet-v3/inception_v3_2016_08_28_frozen.pb

sudo python3 $tf_pb_to_tb  --model_dir $tf_gnv3 --log_dir ~/Documents/tfmodel/TensorBoard/
tensorboard --logdir=/home/mohammad/Documents/tfmodel/TensorBoard/

## (06) How to Download a Deep Learning Based Model
cd /opt/intel/openvino/deployment_tools/tools/model_downloader
python3 downloader.py -h
python3 downloader.py --print_all
sudo python3 downloader.py --all

# optimizing mrcnn50 model from .pb to .bin & .xml
export tf_mrcnn50=/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/public/mask_rcnn_resnet50_atrous_coco/mask_rcnn_resnet50_atrous_coco_2018_01_28
sudo python3 mo_tf.py --input_model $tf_mrcnn50/frozen_inference_graph.pb --output_dir /home/mohammad/Documents/tfmodel/mrcnn50 --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/mask_rcnn_support.json --tensorflow_object_detection_api_pipeline_config $tf_mrcnn50/pipeline.config

### DRAFT ###
# cocostuff model
export caffe_model=/home/mohammad/Documents/caffemodel/sceneparsing/
export caffe_IRmodel=/home/mohammad/Documents/caffemodel/sceneparsing/IRmodel
sudo python3 mo_caffe.py --input_model /home/mohammad/Documents/caffemodel/sceneparsing/DilatedNet.caffemodel --output_dir $caffe_IRmodel