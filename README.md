# YOLO-in-pytorch
This is my self coded version of the 'You only look once' architecture in pytorch 

The constituents of the YOLO folder and their applications are:
<br>
1.  **darknet.py** This file reads the cfg file in cfg folder and creates the network modules. It defines the Darknet class which has functions to carry the forward pass of the YOLO model and load weights.

2.  **util.py** This file contains helper functions that assist darknet.py and detect.py.

3.  **detect.py** This file is responsible for parsing command line arguments and calling suitable
functions to detect dogs in the test images, draw bounding boxes if they exist and save the
results in det folder.

4.  **cfg/yolov3.cfg** This file is the configuration file containing the network layout that is to be
used for detection

5.  **data/coco.names** The data file. It contains the names of all classes in the COCO dataset.

6.  **test_images** folder Contains images to be tested on

7.  **det** folder Contains the test images with bounding boxes (network output)

8.  **yolov3.weights** Weight file

9.  **Pallets** The colours used for bounding boxes.


I have used the following set of tutorials for this:
https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
