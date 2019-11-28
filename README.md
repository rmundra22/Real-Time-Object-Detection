# Real-Time-Object-Detection
Using pre-trained MobileNet SSD for Real Time Multi-Object Detection

> There are two type of deep neural networks here. 
  A. Base network 
  B. detection network 

MobileNet, VGG-Net, LeNet and all of them are base networks. Base network provide high level features for classification or detection. For classification we add a fully connected layer at the end of this networks. But if we remove fully connected layer and replace it with detection networks, like SSD, Faster R-CNN, and so on. In fact, SSD use of last convolutional layer on base networks for detection task. MobileNet just like other base networks use of convolution to produce high level features.

> This module loads pre-trained model for multiclass object detection from a video feed. Besides MobileNet-SDD other architectures can also be used for the same.

   1. GoogleLeNet
   2. YOLO
   3. SqueezeNet
   4. Faster R-CNN
   5. ResNet
   
> The above code establish the following arguments:

  – video: Path file video.
  – prototxt: Network file is .prototxt
  – weights: Network weights file is .caffemodel
  – thr: Confidence threshold.
  
> Runnning this file
  1. Download the pretrained model files namely 'MobileNetSSD_deploy.prototxt' and 'MobileNetSSD_deploy.caffemodel' files.
  2. Check if the video camera in your device works properly. Code switches it on automatically once the code starts.
  3. Use the below commond to execute the python file -
  
 >> python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel



