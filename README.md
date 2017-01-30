# Self-Driving Car Engineer Nanodegree 
## Project III : Behavioral Cloning - Navigating a Car in a Simulator


[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Overview

The objective of this project is to clone human driving behavior using a Deep Neural Network. In order to achieve this, we are going to use a simple Car Simulator. During the training phase, we navigate our car inside the simulator using the keyboard. While we navigating the car the simulator records training images and respective steering angles. Then we use those recorded data to train our neural network. Trained model was tested on two tracks, namely training track and validation track. Following two animations show the performance of our final model in both training and validation tracks.

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:
- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [TensorFlow](http://tensorflow.org)
- [Pandas](http://pandas.pydata.org/)
- [OpenCV](http://opencv.org/)
- [Matplotlib](http://matplotlib.org/) (Optional)
- [Jupyter](http://jupyter.org/) (Optional)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`


### Dataset
Model was trained on two datasets: 

- Collected by driving in train mode in simulator (provided by Udacity)
- Data made available by Udacity [(Download the dataset)](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). This is the sample data for track 1 which contains images (160x320). And put it into folder "data/IMG"

Track 1 | Track 2
------------|---------------
![training_img](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/track1_b.png) | ![validation_img](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/track2_a.png)



### Dataset Statistics
The dataset consists of 8036 images . The training track contains a lot of shallow turns and straight road segments. Hence, the majority of the recorded steering angles are zeros (4361 center steering images) while the rest are 1775 left steering images and 1900 right steering images. Therefore, preprocessing images and respective steering angles are necessary in order to generalize the training model for unseen tracks such as our validation track.

Left| Center | Right
----|--------|-------
![left](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/left_2016_12_01_13_34_37_857.jpg) | ![center](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/center_2016_12_01_13_34_37_857.jpg) | ![right](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/right_2016_12_01_13_34_37_857.jpg)

### Dataset Augmentation
8036 images in Udacity dataset is not enough for a full train, especially for recovery and for generalization for track 2. I applied following augmentation techniques:


- Randomly change brightness

- Transition horizontally. Without transition recovery doesn't work in my model.

- Crop, to reduce non-valued information

- Random shadow, for the second track

- Flip

Brightness Augmentaion | Shadow Augmentaion
------------|---------------
![training_img](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/center_bright.png) | ![validation_img](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/center_shadow.png)




###  Model Architecture
CNN architecture in this project was inspired by [NVIDIA's End to End Learning for Self-Driving Cars paper](https://arxiv.org/pdf/1604.07316v1.pdf)

       ___________________________________________________________________________________________________
        Layer (type)                     Output Shape          Param #     Connected to                     
        ====================================================================================================
        lambda_8 (Lambda)                (None, 103, 320, 3)   0           lambda_input_8[0][0]             
        ____________________________________________________________________________________________________
        convolution2d_43 (Convolution2D) (None, 52, 160, 3)    228         lambda_8[0][0]                   
        ____________________________________________________________________________________________________
        activation_71 (Activation)       (None, 52, 160, 3)    0           convolution2d_43[0][0]           
        ____________________________________________________________________________________________________
        convolution2d_44 (Convolution2D) (None, 26, 80, 24)    1824        activation_71[0][0]              
        ____________________________________________________________________________________________________
        activation_72 (Activation)       (None, 26, 80, 24)    0           convolution2d_44[0][0]           
        ____________________________________________________________________________________________________
        maxpooling2d_36 (MaxPooling2D)   (None, 26, 80, 24)    0           activation_72[0][0]              
        ____________________________________________________________________________________________________
        dropout_36 (Dropout)             (None, 26, 80, 24)    0           maxpooling2d_36[0][0]            
        ____________________________________________________________________________________________________
        convolution2d_45 (Convolution2D) (None, 13, 40, 36)    21636       dropout_36[0][0]                 
        ____________________________________________________________________________________________________
        activation_73 (Activation)       (None, 13, 40, 36)    0           convolution2d_45[0][0]           
        ____________________________________________________________________________________________________
        maxpooling2d_37 (MaxPooling2D)   (None, 13, 40, 36)    0           activation_73[0][0]              
        ____________________________________________________________________________________________________
        dropout_37 (Dropout)             (None, 13, 40, 36)    0           maxpooling2d_37[0][0]            
        ____________________________________________________________________________________________________
        convolution2d_46 (Convolution2D) (None, 7, 20, 48)     43248       dropout_37[0][0]                 
        ____________________________________________________________________________________________________
        activation_74 (Activation)       (None, 7, 20, 48)     0           convolution2d_46[0][0]           
        ____________________________________________________________________________________________________
        maxpooling2d_38 (MaxPooling2D)   (None, 7, 20, 48)     0           activation_74[0][0]              
        ____________________________________________________________________________________________________
        dropout_38 (Dropout)             (None, 7, 20, 48)     0           maxpooling2d_38[0][0]            
        ____________________________________________________________________________________________________
        convolution2d_47 (Convolution2D) (None, 7, 20, 64)     27712       dropout_38[0][0]                 
        ____________________________________________________________________________________________________
        activation_75 (Activation)       (None, 7, 20, 64)     0           convolution2d_47[0][0]           
        ____________________________________________________________________________________________________
        maxpooling2d_39 (MaxPooling2D)   (None, 7, 20, 64)     0           activation_75[0][0]              
        ____________________________________________________________________________________________________
        dropout_39 (Dropout)             (None, 7, 20, 64)     0           maxpooling2d_39[0][0]            
        ____________________________________________________________________________________________________
        convolution2d_48 (Convolution2D) (None, 7, 20, 64)     36928       dropout_39[0][0]                 
        ____________________________________________________________________________________________________
        activation_76 (Activation)       (None, 7, 20, 64)     0           convolution2d_48[0][0]           
        ____________________________________________________________________________________________________
        maxpooling2d_40 (MaxPooling2D)   (None, 7, 20, 64)     0           activation_76[0][0]              
        ____________________________________________________________________________________________________
        dropout_40 (Dropout)             (None, 7, 20, 64)     0           maxpooling2d_40[0][0]            
        ____________________________________________________________________________________________________
        flatten_8 (Flatten)              (None, 8960)          0           dropout_40[0][0]                 
        ____________________________________________________________________________________________________
        dense_36 (Dense)                 (None, 1164)          10430604    flatten_8[0][0]                  
        ____________________________________________________________________________________________________
        activation_77 (Activation)       (None, 1164)          0           dense_36[0][0]                   
        ____________________________________________________________________________________________________
        dense_37 (Dense)                 (None, 100)           116500      activation_77[0][0]              
        ____________________________________________________________________________________________________
        activation_78 (Activation)       (None, 100)           0           dense_37[0][0]                   
        ____________________________________________________________________________________________________
        dense_38 (Dense)                 (None, 50)            5050        activation_78[0][0]              
        ____________________________________________________________________________________________________
        activation_79 (Activation)       (None, 50)            0           dense_38[0][0]                   
        ____________________________________________________________________________________________________
        dense_39 (Dense)                 (None, 10)            510         activation_79[0][0]              
        ____________________________________________________________________________________________________
        activation_80 (Activation)       (None, 10)            0           dense_39[0][0]                   
        ____________________________________________________________________________________________________
        dense_40 (Dense)                 (None, 1)             11          activation_80[0][0]              

        Total params: 10,684,251
        Trainable params: 10,684,251
        Non-trainable params: 0


### Hyperparameter

- batch_size = 512

- number_of_epochs = 8

- number_of_samples_per_epoch = 20032

- number_of_validation_samples = 6400



### Training
`fit_generator` API of the Keras library is used to train the deep CNN model for all augmented images.

We created two generators namely:
* `training_generator = myGenerator()`
* `validation_generator = myGenerator()` 

Batch size of both `train_gen` and `validation_gen` was 64. We used 20032 images per training epoch. It is to be noted that these images are generated on the fly using the document processing pipeline described above. In addition to that, we used 6400 images (also generated on the fly) for validation. We used `Adam` optimizer with `1e-4` learning rate. Finally, when it comes to the number of training epochs we tried several possibilities such as `5`, `8`, `1`0, `2`5 and `50`. However, `8` works well on both training and validation tracks.


### How to Run the Model

This repository comes with trained model which you can directly test using the following command.

- Save trained model (model.json and model.h5) 

- Start simulator in autonomous mode

- Start server: `python drive.py model.json`


### Results

Training Track 1 | Validation Track 2
------------|---------------
![training_img](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/train_track1.png) | ![validation_img](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/train_track2.png)



