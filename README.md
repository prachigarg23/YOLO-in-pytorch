# YOLO-in-pytorch
This is my self coded version of the 'You only look once' architecture in pytorch 

The constituents of the YOLO folder and their applications are:
<br>
1. **darknet.py** This file reads the cfg file in cfg folder and creates the network modules. It defines the Darknet class which has functions to carry the forward pass of the YOLO model and load weights.
<br>
2. **util.py** This file contains helper functions that assist darknet.py and detect.py.
<br>
3. **detect.py** This file is responsible for parsing command line arguments and calling suitable
functions to detect dogs in the test images, draw bounding boxes if they exist and save the
results in det folder.
<br>
4. **cfg/yolov3.cfg** This file is the configuration file containing the network layout that is to be
used for detection
<br>
5. **data/coco.names** The data file. It contains the names of all classes in the COCO dataset.
<br>
6. **test_images** folder Contains images to be tested on
<br>
7. **det** folder Contains the test images with bounding boxes (network output)
<br>
8. **yolov3.weights** Weight file
<br>
9. **Pallets** The colours used for bounding boxes.
