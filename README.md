# ECE570Final
Object Detection and Instance Segmentation with COCO Dataset
This project implements an object detection and instance segmentation model using a two-stage detection method inspired by the D2Det approach. The model is designed to handle precise localization and accurate classification of objects on a subset of the COCO dataset. It includes data loading, model training, and evaluation scripts, as well as code for visualizing predictions.

Table of Contents
Overview
Features
Requirements
Setup Instructions
Usage
1. Dataset Preparation
2. Training the Model
3. Evaluating the Model
Customization
Limitations and Future Work
Overview
The project implements a two-stage object detection pipeline with a focus on dense localization regression and improved feature pooling for classification. It uses a subset of the COCO dataset and includes functionality for splitting the dataset into train and test subsets, training the model, and evaluating it on unseen data.

Features
Customizable data loading with torchvision.datasets.CocoDetection.
Custom train-test dataset splitting and sampling for subset training.
Model structure based on torchvision with RoI Align and Feature Pyramid Network (FPN).
Training and evaluation routines.
Basic visualization for model predictions.
Requirements
Hardware: It is recommended to have a GPU for training.
Software: The following Python packages are required (they are also listed in requirements.txt):
torch
torchvision
matplotlib
numpy
tqdm
pycocotools
Setup Instructions

cd repo-name
Install dependencies: Install all necessary packages listed in requirements.txt:

bash
Copy code
pip install -r requirements.txt
Directory Structure: Ensure your directory is structured as follows:


coco_mini: Folder containing the COCO dataset images and annotations.
train2017: Folder containing the training images.
annotations: Folder containing the JSON annotation file (instances_train2017.json).
Usage
1. Dataset Preparation
Download a subset of the COCO dataset if not already available. Place images in the train2017 directory and annotations in the annotations directory, with the structure as shown above.

2. Training the Model
To train the model, use the following command:

bash
Copy code
python main.py --train
This command will:

Load the dataset.
Split it into training and testing subsets.
Train the model on the specified dataset subset.
3. Evaluating the Model
To evaluate the trained model, use:

bash
Copy code
python main.py --eval
This command will:

Load the test dataset.
Evaluate the model on the test data.
Display key metrics such as Mean Average Precision (mAP).
Optional Arguments
--batch_size: Set the batch size for training/evaluation.
--epochs: Specify the number of epochs for training.
--learning_rate: Set the learning rate for optimization.
Example
bash
Copy code
python main.py --train --batch_size 8 --epochs 10 --learning_rate 0.001
Customization
Changing Backbone: Modify the backbone in the model architecture to experiment with different feature extractors.
Data Augmentation: Customize data augmentations by editing the transform function in the code.
Subset Size: Adjust the subset size of the dataset in the data loading section.
Limitations and Future Work
This implementation was trained on a small subset of the COCO dataset due to hardware limitations. Consequently, the model may not generalize as well as state-of-the-art methods trained on the full COCO dataset. Future improvements could include:

Using a larger dataset or a full COCO dataset for improved accuracy.
Optimizing the model structure for efficient inference.
Experimenting with more advanced feature pooling and localization techniques.
