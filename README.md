# LandmarksClassification

# Description
A CNN training pipeline that leverages transfer learning by applying feature representations learned from a related source task to enhance performance in the target task of European landmark classification.

Completed as a course project for EECS 445 Intro to Machine Learning in Winter 2024 at the University of Michigan.

# Set up
Python 3.11
1. Clone the project folder
2. Run `pip install -r requirements.txt`

# Training Procedure
1. Place the source data into the data/images folder (for training images) and the landmarks.csv file (for labels).
2. Run data augmentation `python augment_data.py data/landmarks.csv data`
3. Run `python train_source.py`
4. Run `train_target.py`
