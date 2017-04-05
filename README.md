# dog_image_classification
Classifies 120 different breeds of dogs using convolutional neural network and tensorflow

#instructions to use:
1. Download the dataset from the link. http://vision.stanford.edu/aditya86/ImageNetDogs/
2. Mention path to Annotation folder and Image folder in line number 6 and 34 of cropImage.py script
3. Also mention the directory at line number 11 where cropped and reshaped images will be saved
4. run the cropImage.py script and this will crop and reshape all images as mentioned in corresponding annotation files
5. Now run the cnn_dog.py script and mention the path to cropped images in line number 7.
6. this script will divide all image dataset in training and testing data in ratio of 4:1
