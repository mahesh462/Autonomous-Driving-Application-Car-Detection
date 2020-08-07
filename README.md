# Autonomous-Driving-Application-Car-Detection
Object detection using YOLO model.

Input images are labelled by drawing bounding boxes around every car. Here's anexample:
<p align = 'center'>
  <img src = '/nb_images/box_label.png'>
</p>

If we have 80 classes which the object detector have to recognize, the class label c is represented as 80 dimensional vector one component of which is one and rest as 0.

## YOLO

"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

# Model details
## Inputs and Outputs
- The input is a batch of images, and each image has the shape (m, 608, 608, 3).
- The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers  (pc,bx,by,bh,bw,c)  as explained above. If you expand  c  into an 80-dimensional vector, each bounding box is then represented by 85 numbers.

## Anchor Boxes
- Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes. 5 anchor boxes are chosen and stored in the file './model_data/yolo_anchors.txt'.
- The dimension for anchor boxes is the second to last dimension in the encoding:  (m,n<sub>H</sub>,n<sub>W</sub>,anchors,classes).
- The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

## Encoding
<p align = 'center'>
  <img src = '/nb_images/architecture.png'>
</p>
If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.


Since we are using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.
For simplicity, we will flatten the last two last dimensions of the shape (19, 19, 5, 85) encoding. So the output of the Deep CNN is (19, 19, 425).
<p align = 'center'>
  <img src = '/nb_images/flatten.png'>
</p>

## Class Score
Now, for each box (of each cell) we will compute the following element-wise product and extract a probability that the box contains a certain class.
The class score is  score<sub>c,i</sub>=p<sub>c</sub>×c<sub>i</sub> : the probability that there is an object  p<sub>c</sub>  times the probability that the object is a certain class  c<sub>i</sub>.
<p align = 'center'>
  <img src = '/nb_images/probability_extraction.png'>
</p>

## Visualizing Classes
Here's one way to visualize what YOLO is predicting on an image:
- For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across the 80 classes, one maximum for each of the 5 anchor boxes).
- Color that grid cell according to what object that grid cell considers the most likely.
<p align = 'center'>
  <img src = '/nb_images/proba_map.png'>
</p>
Note that this visualization isn't a core part of the YOLO algorithm itself for making predictions; it's just a nice way of visualizing an intermediate result of the algorithm.

## Visualizing Bounding Boxes
Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this:
<p align = 'center'>
  <img src = '/nb_images/anchor_map.png'>
</p>

## Non-max Suppression
In the figure above, we plotted only boxes for which the model had assigned a high probability, but this is still too many boxes. We'd like to reduce the algorithm's output to a much smaller number of detected objects.


To do so, we'll use non-max suppression. Specifically, we'll carry out these steps:
- Get rid of boxes with a low score (meaning, the box is not very confident about detecting a class; either due to the low probability of any object, or low probability of this particular class).
- Select only one box when several boxes overlap with each other and detect the same object.
<p align = 'center'>
  <img src = '/nb_images/non-max-suppression.png'>
</p>

Non-max suppression uses the very important function called "Intersection over Union", or IoU.
<p align = 'center'>
  <img src = '/nb_images/iou.png'>
</p>


# Filtering with a Threshold on Class Scores
We are going to first apply a filter by thresholding. We would like to get rid of any box for which the class "score" is less than a chosen threshold.


The model gives a total of 19x19x5x85 numbers, with each box described by 85 numbers. It is convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:
- box_confidence: tensor of shape  (19×19,5,1)  containing  pc  (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
- boxes: tensor of shape  (19×19,5,4)  containing the midpoint and dimensions  (bx,by,bh,bw)  for each of the 5 boxes in each cell.
- box_class_probs: tensor of shape  (19×19,5,80)  containing the "class probabilities"  (c1,c2,...c80)  for each of the 80 classes for each of the 5 boxes per cell.
