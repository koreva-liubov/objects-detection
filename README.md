# Object Detection Using Tensorflow

## Get Tensorflow Object detection API 

Clone models from Tensorflow Model Garden:

```
git clone https://github.com/tensorflow/models
```

## Install dependencies

Follow the instructions to install Anaconda if it's not installed on your device yet:
```
https://docs.anaconda.com/anaconda/install/
```

TensorFlow or TensorFlow-GPU if you have a GPU on your device
```
pip install tensorflow
```
```
pip install tensorflow-gpu
```
Open-CV:
```
pip install opencv-python
```

Some other dependencies:
```
pip install jupyter lxml matplotlib pillow Cython tf_slim contextlib2 
```

Download and extract Protobuf (mechanism for serializing structured data) to `models/research` folder:
```
https://github.com/protocolbuffers/protobuf/releases
```

## Compile protocol buffer description files
Generate Python code from a .proto file

```
cd models/research
protoc --python_out=. object_detection/protos/*.proto
ls object_detection/protos/*.proto
```
## Run object detection code
```
cp object-detection.py models/research
cd models/research
python object-detection.py
```
# Object Detection Using OpenCV

## Get dependencies
Download the pre-trained YOLO weight file (237 MB): https://pjreddie.com/media/files/yolov3.weights

## Run object detection code
```
cd opencv
python cv-detection.py
```
