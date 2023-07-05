# ProjectSiemens

# Object Recognition, Classification, and Detection Project 

This project focuses on object recognition, classification, and detection. It is an educational endeavor aimed at understanding how these processes work, starting from simple neural networks and potentially advancing to larger pre-trained models. 

Our main performance metric is not the ultimate performance of the model, but the optimization of learning throughout the process. As such, the loss function is used to assess and refine our learning approach, rather than the model's performance.

## Installation

To get started with this project, you'll need to install the required packages. Assuming you're using pip, you can do this by running:

```bash
pip install -r requirements.txt

 Database

The project uses the MVTec Screws dataset for training and testing the model. Here's a brief description of the dataset:

Overview

MVTec Screws contains 384 images of 13 different types of screws and nuts on a wooden background. The objects are labeled by oriented bounding boxes and their respective category. Overall, there are 4426 such annotations.

The data has been split in a way that approximately 70% of the instances of each category are within the training split, and 15% each in the validation and test splits.

 Structure

- `images`: This folder contains all the screw images.
- `mvtec_screws.json`: This file contains the annotations for all images in COCO format.
- `mvtec_screws_train/val/test.json`: These files contain the example splits as mentioned above, in COCO format.
- `mvtec_screws.hdict`: This file contains the DLDataset unsplitted.
- `mvtec_screws_split.hdict`: This file contains the DLDataset with splits.

Usage

The .hdict files can be used within HALCON by reading them, and the image path has to be set to the location of the images folder. For usage within HALCON, no conversion is needed as the format is also used within the deep learning based object detection of HALCON.

Format

MVTec screws is a dataset for oriented box detection, using a format that is very similar to that of the COCO dataset. However, it includes an additional parameter to store the orientation of each box. Each box contains 5 parameters (row, col, width, height, phi), where:
- 'row' is the subpixel-precise center row (vertical axis of the coordinate system) of the box.
- 'col' is the subpixel-precise center column (horizontal axis of the coordinate system) of the box.
- 'width' is the subpixel-precise width of the box, i.e., the length of the box parallel to the orientation of the box.
- 'height' is the subpixel-precise width of the box, i.e., the length of the box perpendicular to the orientation of the box.
- 'phi' is the orientation of the box in radians, given in a mathematically positive sense and with respect to the horizontal (column) image axis.
