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


Usage
1. Dataset Preparation
Download a subset of the COCO dataset if not already available. Place images in the train2017 directory and annotations in the annotations directory, with the structure as shown above.

Load the dataset.
Split it into training and testing subsets.

Train the model on the specified dataset subset

3. Evaluating the Model
To evaluate the trained model

Load the test dataset.

Evaluate the model on the test data.

Display key metrics such as Mean Average Precision (mAP).

Using a larger dataset or a full COCO dataset for improved accuracy.
Optimizing the model structure for efficient inference.
Experimenting with more advanced feature pooling and localization techniques.
