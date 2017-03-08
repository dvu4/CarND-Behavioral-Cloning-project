# Self-Driving Car Engineer Nanodegree 
## Project III : Behavioral Cloning - Navigating a Car in a Simulator



[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Overview

The objective of this project is to clone human driving behavior using a Deep Neural Network. In order to achieve this, we are going to use a simple Car Simulator. During the training phase, we navigate our car inside the simulator using the keyboard. While we navigating the car the simulator records training images and respective steering angles. Then we use those recorded data to train our neural network. Trained model was tested on two tracks, namely training track and validation track. Following two animations show the performance of our final model in both training and validation tracks.

This repository contains 5 files for the Behavioral Cloning Project: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)
* writeup_template.md

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

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




This README file describes how to output the video in the "Details About Files In This Directory" section.


## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).






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


Shown below is a histogram of the steering angles recorded while driving in the middle of the road for a few laps. This is the data used for training. The left-right skew can be eliminated by flipping images and steering angles simultaneously. However, even after balancing left and right angles most of the time the steering angle during normal driving still introduces a bias towards driving straight especially when the car needs to turn sharply.

![training_img](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/stastitical.png)


### Dataset Augmentation
8036 images in Udacity dataset is not enough for a full train, especially for recovery and for generalization for track 1, 2. To augment the dataset, certain random transformations were applied to the provided images. These include:

- Using left/right cameras instead of center with steering offsets of 0.2/-0.2

- Randomly change brightness

- Transition horizontally. Without transition recovery doesn't work in my model.

- Crop, to reduce non-valued information

- Random shadow, for the second track

- Flipping vertically (this was originally done randomly, then done epoch-by-epoch)


Brightness Augmentaion | Shadow Augmentaion
------------|---------------
![training_img](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/center_bright.png) | ![validation_img](https://raw.githubusercontent.com/dvu4/CarND-Behavioral-Cloning-project/master/data/center_shadow.png)




###  Model Architecture
CNN architecture in this project was inspired by [NVIDIA's End to End Learning for Self-Driving Cars paper](https://arxiv.org/pdf/1604.07316v1.pdf)

      LAYER: Normalization                  (None, 200, 200, 3)
       ____________________________________________________________________________________________________
       Layer (type)                     Output Shape          Param #     Connected to                     
       ====================================================================================================
       Normalization (Lambda)           (None, 200, 200, 3)   0           lambda_input_46[0][0]            
       ____________________________________________________________________________________________________
       convolution2d_201 (Convolution2D (None, 100, 100, 24)  672         Normalization[0][0]              
       ____________________________________________________________________________________________________
       maxpooling2d_63 (MaxPooling2D)   (None, 50, 50, 24)    0           convolution2d_201[0][0]          
       ____________________________________________________________________________________________________
       convolution2d_202 (Convolution2D (None, 25, 25, 36)    7812        maxpooling2d_63[0][0]            
       ____________________________________________________________________________________________________
       maxpooling2d_64 (MaxPooling2D)   (None, 12, 12, 36)    0           convolution2d_202[0][0]          
       ____________________________________________________________________________________________________
       convolution2d_203 (Convolution2D (None, 12, 12, 48)    15600       maxpooling2d_64[0][0]            
       ____________________________________________________________________________________________________
       maxpooling2d_65 (MaxPooling2D)   (None, 6, 6, 48)      0           convolution2d_203[0][0]          
       ____________________________________________________________________________________________________
       convolution2d_204 (Convolution2D (None, 6, 6, 64)      27712       maxpooling2d_65[0][0]            
       ____________________________________________________________________________________________________
       dropout_241 (Dropout)            (None, 6, 6, 64)      0           convolution2d_204[0][0]          
       ____________________________________________________________________________________________________
       flatten_42 (Flatten)             (None, 2304)          0           dropout_241[0][0]                
       ____________________________________________________________________________________________________
       dropout_242 (Dropout)            (None, 2304)          0           flatten_42[0][0]                 
       ____________________________________________________________________________________________________
       dense_129 (Dense)                (None, 512)           1180160     dropout_242[0][0]                
       ____________________________________________________________________________________________________
       leakyrelu_62 (LeakyReLU)         (None, 512)           0           dense_129[0][0]                  
       ____________________________________________________________________________________________________
       dropout_243 (Dropout)            (None, 512)           0           leakyrelu_62[0][0]               
       ____________________________________________________________________________________________________
       dense_130 (Dense)                (None, 10)            5130        dropout_243[0][0]                
       ____________________________________________________________________________________________________
       leakyrelu_63 (LeakyReLU)         (None, 10)            0           dense_130[0][0]                  
       ____________________________________________________________________________________________________
       dense_131 (Dense)                (None, 1)             11          leakyrelu_63[0][0]               
       ====================================================================================================
       Total params: 1,237,097
       Trainable params: 1,237,097
       Non-trainable params: 0
### Hyperparameter

- batch_size = 1000

- number_of_epochs = 5

- number_of_samples_per_epoch = 30000

- number_of_validation_samples = 5000



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


