# Reading Cattle Ear Tags and Drinking Behaviour with Computer Vision on Jetson AGX Xavier with Deepstream
This project uses a two-step system to detect and read cattle ear tags using deepstream on a jetson device:
- **Step 1**: use custom **Tiny Yolov5** model embedded in deepstream to locate cattle ear tags and drinking cattle from webcam or video images.
- **Step 2**: use deepstream tracking layer to assign a unique tracking ID to a located tag according to past frames.
- **Step 3**: for tags with new trackingIDs or unsure previous predictions, use a fine-tuned **TRBA** (TPS-ResNet-BiLSTM-Attn) model to read ear tags and save the reading for future frames. [1]
<img src="./figures/example.png" width="650" title="example">

This repository contains code that is meant to be run on Nvidia Jetson devices. For code that can run on non-jetson devices please see [here](https://github.com/msmink01/computer_vision_cattle_tag_reading).

### Current Possibilities
| Input type | Possible output type | Makefile command to use | Python file used |
| --- | --- | --- | --- |
| Video (h264, mp4) | Screen | make | *mc-test4-read-tags-faster-one-src.py* |
| Webcam (v4l2) | Screen | make cam | *mc-test5-read-tags-faster-usbcam* |


***
## Getting Started
### External Requirements:
1. A Jetson device capable of being flashed with Jetpack Version 5.1 and Deepstream 6.1. 
2. The means of flashing the Jetson device, either an SD card image or a host PC running Linux, depending on your jetson device. For more information please visit [Nvidia's flashing tutorials](https://github.com/dusty-nv/jetson-inference/blob/master/docs/jetpack-setup-2.md).
3. Sample inputs (either): 
   - A V4L2 usb camera. (For use of other types of webcams, adjustments must be made to deepstream video converters in py files)
   - Sample cattle videos in h264 or mp4 format
4. Custom model files for the tag detector and tag reader. Please email me for these (msmink01 AT gmail DOT com).

### Setting up your jetson environment
1. **Set up your Jetson device**: \
   a. Flash your Jetson device with Jetpack 5.1 and Deepstream 6.1 according to [Nvidia's tutorials](https://github.com/dusty-nv/jetson-inference/blob/master/docs/jetpack-setup-2.md). 
   
   b. Open your Jetson terminal and upgrade your pip environment
      - ``$ sudo apt-get update``
      - ``$ sudo apt-get upgrade``
      - ``$ sudo apt install python3-pip``
   
   c. Create a virtual environment
      - ``$ python3 -m venv myenv``
   
   d. Activate your virtual environment
      - ``$ source myenv/bin/activate``\
  
   e. Install gstreamer dependencies needed later
      - ``$ sudo apt install libssl1.1 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev``



2. **Install Yolov5 Capabilities in your jetson**: \
Steps taken from [yolov5's jetson tutorial](https://github.com/ultralytics/yolov5/issues/9627)
   
   a. Clone the [yolov5 repository](https://github.com/ultralytics/yolov5) onto your jetson and go into it
      - ``$ git clone https://github.com/ultralytics/yolov5.git``
      - ``$ cd yolov5``
   
   b. Edit *requirements.txt* so the lines regarding torch and torchvision are commented out.
      - *requirements.txt* should now contain ``# torch>=1.7.0`` and ``# torchvision>=0.8.1`` instead of ``torch>=1.7.0`` and ``torchvision>=0.8.1``
   
   c. Install a dependency (make sure you are in your virtual environment).
      - ``$ sudo apt install -y libfreetype6-dev``
   
   d. Install the updated *requirements.txt* (make sure you are in your virtual environment).
      - ``$ pip3 install -r requirements.txt``
      

3. **Install torch and torchvision in your jetson**: \
Steps taken from [yolov5's jetson tutorial](https://github.com/ultralytics/yolov5/issues/9627)
   
   a. Install torch v1.12.0 using (nvidia's jetson-specific files)[https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048].
      - ``$ cd ~``
      - ``$ sudo apt-get install -y libopenblas-base libopenmpi-dev``
      - ``$ wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl -O torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl``
      - ``$ pip3 install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl``
   
   b. Install torchvision v0.13.0 from [torchvision's github](https://github.com/pytorch/vision) (torchvision version is dependant on your torch version)
      - ``$ sudo apt-get install -y libjpeg-dev zlib1g-dev``
      - ``$ git clone --branch v0.13.0 https://github.com/pytorch/vision torchvision``
      - ``$ cd torchvision``
      - ``$ python3 setup.py install``; if run into admin errors sudo this command
      
   c. Check your installation of these packages using ``$ pip3 list``. If successful you should see the correct versions of torch/torchvision listed.
   

***
### References
[1] [J. Baek, G. Kim, J. Lee, S. Park, D. Han, S. Yun, S. J. Oh, and H. Lee. What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis. International Conference on Computer Vision (ICCV). 2019.](https://github.com/clovaai/deep-text-recognition-benchmark)\
[2] [Yolov5 v6.2 by Ultralytics](https://github.com/ultralytics/yolov5)
