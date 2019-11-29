# Real-Time-Object-Detection
### Using pre-trained MobileNet SSD for Real Time Multi-Class-Object Detection


> There are two type of deep neural networks here. 
  1. Base network 
  2. detection network 

MobileNet, VGG-Net, LeNet and all of them are base networks. Base network provide high level features for classification or detection. For classification we add a fully connected layer at the end of this networks. But if we remove fully connected layer and replace it with detection networks, like SSD, Faster R-CNN, and so on. In fact, SSD use of last convolutional layer on base networks for detection task. MobileNet just like other base networks use of convolution to produce high level features.

> This module loads pre-trained model for multiclass object detection from a video feed. Besides MobileNet-SDD other architectures can also be used for the same.
   1. GoogleLeNet
   2. YOLO
   3. SqueezeNet
   4. Faster R-CNN
   5. ResNet
   
> The above code establish the following arguments:
  1. video: Path file video.
  2. prototxt: Network file is .prototxt
  3. weights: Network weights file is .caffemodel
  4. thr: Confidence threshold.

> Runnning this file
  1. Download the pretrained model files namely 'MobileNetSSD_deploy.prototxt' and 'MobileNetSSD_deploy.caffemodel' files.
  2. Check if the video camera in your device works properly. Code switches it on automatically once the code starts.
  3. Use the below commond to execute the python file:- python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

> My Experiments

![Sample - Experiment_1](test_sample_videos/sample_1.gif)

In the above gif we can observe that we are able to detect multiple objects in real-time using laptop's video cam. But it may happen sometimes that the model fail to detect an object class or it may also happen that occuluded objects are neither detected/correctly-classified. It's a pretty basic model to give a good feel of object detection without exclusively training a model with some personilized dataset. Retraining the model with some personalized data may help to give better results.

![Sample - Experiment_2](test_sample_videos/sample_2.gif)

 
### Single Shot Detection (SSD)

In simple words it means you take a single look at an image to propose object-detections even if we are require to detect multiple objects within the image. The difference between the SSD and Regional Proposal Network (RPN) based approaches such as R-CNN series is that they are a 2 stage algorithms. In other words it is that they need two shots, one for generating region proposals and another one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches. Because of this feature of the SSD based approach we can get real time detections.

> SSD300 achieves 74.3% mAP at 59 FPS while SSD500 achieves 76.9% mAP at 22 FPS, which outperforms Faster R-CNN (73.2% mAP at 7 FPS) and YOLOv1 (63.4% mAP at 45 FPS). This Single Shot Multibox Detector is Cited by 6083 when I am writing this. Some Crazy Object Detection Results on MS COCO dataset can be visualize using the image below :-

![Crazy Object Detections](https://miro.medium.com/max/1309/1*4BLx59c0GBy1v7hPL3nn2g.png)


