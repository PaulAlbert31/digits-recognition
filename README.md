# Digits recognition

This is a personal project i work on during my free time. Feel free to try it yourself, i will add the instruction further down. I use the C++ language for now but i might switch to python in the future.
The document Base.h is a courtesy of Hervé Frezza-Buet : http://www.metz.supelec.fr/metz/personnel/frezza/.
The neural network approach is based on a tutorial by Jeremy Fix : http://www.metz.supelec.fr/metz/personnel/fix_jer/.

# What's new ?

  - I am currently working on dataset augmentation to better the convolutional network's performances.


Yet to be done:
  - ~~Unsupervised classification using online k-means.~~
    - ~~Improve the online k-means by using Kohonen maps.~~
    - ~~Improve the representation of the digit to make it robust to scaling and rotation.~~
    - ~~Improve k-means classification by blurring th images.~~
    - ~~Regroup all used function in a .h file.~~
- Supervised classification using SVM
- Supervised classification using ensemble methods
- Reiforcement learning using online learners
- Reinforcement learning using batch learners
- Deep learning methods (architectures will be described bellow)
    - ~~Linear classifier.~~
    - ~~Simple convulationnal network.~~
    - ~~Convolutionnal networks with less parameters.~~
    - Dataset augmentation.
    - Probability averaging on several networks.
    - Tests on the CIFAR-100 dataset.

# Want to run it yourself ?

### Kmeans + Kokonen maps
```sh
$ cd digits-recognition
$ g++ -o online-kmeans-kohonen -Wall -ansi -O3 online-kmeans-kohonen.cc
$ ./online-kmeans-kohonen
```
This will create a number of images in the current directory, please set VERBOSE to false if you don't want to.
```cpp
#define VERBOSE true
```
There is now other options :
  - BLUR will blur the images if set to TRUE.
  - BFACTOR is the factor you want to use to blur the images (should be < 1).
  - TEST will genrerate 100 predictions of the classification prototype and stroge them in files "imagette-xxx.ppm" if set to TRUE. The label will be written in the console you type ./online-kmeans-kohonen in.
  - REDUCENOISE will attempt to clear the final classification prototype (not very effective yet) if set to TRUE.
    
To make a short film out of it : (only if the images were generated)

```sh
$ avconv -i kmeans-%06d.ppm -b:v 1M kmeans.avi
```
To delete all images + film :
```sh
$ find . -name 'kmeans*' -delete
```
### Cluster generation for edges points
```sh
$ cd digits-recognition
$ g++ -o shape-context -Wall -ansi -O3 shape-context.cc
$ ./shape-context
```
This will create a number of images in the current directory, please set VERBOSE to false if you don't want to.
```cpp
#define VERBOSE true
```
A "map" file will be generated, containing the cluster data.
To make a short film out of it : (only if the images were generated)

```sh
$ avconv -i context-%06d.ppm -b:v 1M context.avi
```
To delete all images + film :
```sh
$ find . -name 'context*' -delete
```
### Image classification using context clustering
```sh
$ cd digits-recognition
$ g++ -o identifier-context -Wall -ansi -O3 identifier-context.cc
$ ./identifier-context
```
This will create a number of images in the current directory, please set VERBOSE to false if you don't want to.
```cpp
#define VERBOSE true
```
The "map" file will be needed, make sure you have previously clustered the edges points.
To make a short film out of it : (only if the images were generated)
For the frequency cluster
```sh
$ avconv -i frequency-%06d.ppm -b:v 1M frequency.avi
```
For the k means + kohonen cluster of the images 
```sh
$ avconv -i kmeans-%06d.ppm -b:v 1M kmeans.avi
```
To delete all images + film :
```sh
$ find . -name 'frequency*' -delete
$ find . -name 'kmeans*' -delete
```
# Neural networks
The folowing networks are developped on Python3. Tensorboard is used to display usefull informations. I ran the code on my school's culster.
For now they are all trained on the mnist database and use the keras package : https://keras.io/.
I will add graphs displaying performances later, the cluster is currently down due to nvidia updates.

### Linear classifier
The file train_mnist_linear.py contains a linear neural network.
The architecture is simple: 
- An input layer that takes the image reshaped in a 1-dim vector
- A linear dense layer with 10 parameters. Keras automatically adds a constant dimension for offset consideration.
- A softmax activation layer, adapted for classification problems.

I also added a normalization of the input in a second time to better performances.
You can set the following variables to True or False if you wich to normalize the input or see a summary of the model.

```py
#Options (could be optimised for preprocessing)
normalization = True
summary = True
```
To see the Tensorboard results : 
```sh
$ tensorboard --logdir ./logs_linear
```
To run the network : 
```sh
$ ./train_mnist_linear.py
```
You might need to make the file executable : 

```sh
$ chmod u+x train_mnist_linear.py
```
Navigate to http://localhost:6006 to see results (might change see in the console after running $ tensorboard --logdir ./logs_linear
### Linear classifier with two hidden layers
This network is a linear network using two hidden layers. The dropout pourcentages were chosen according to the Srivastava2014 paper recomendations.

The architecture : 
- An input layer that takes the image reshaped in a 1-dim vector
- Two dense layers with relu activation, 256 neurons each and optional dopout layers (50%) to prevent overfiting.
- A linear dense, fully connected layer with 10 parameters.
- A softmax activation layer, adapted for classification.

```py
#Options (could be optimised for preprocessing)
dropout = True
normalization = True
summary = True
```
To see the Tensorboard results : 
```sh
$ tensorboard --logdir ./logs_hidden2
```
To run the network : 
```sh
$ ./train_mnist_hidden2.py
```
You might need to make the file executable : 

```sh
$ chmod u+x train_mnist_hidden2.py
```
### Convolutional network
Convolutional network using three 5x5 kernels with 16 - 32 - 64 filters, and stride=1 to prevent reduction of the representation. The activation layer uses relu a function and a maxpool layer is finnaly to reduce the size of the representation by 2.

The architecture : 
- An input layer that takes 4D Tensors representing the image.
- Optional dropout layer (20%) to prevent overfiting.
- Three conv 5x5 layers with 13 - 32 - 64 filters, relu activation and maxpool layers.
- Two fully connected layer with 128 and 64 neurons, relu activation layer and optional 50% dropout layers.
- A flatten layer to convert 4D Tensors back to 2D Tensors.
- A dense layer with 10 neurons + softmax activation layer, adapted for classification.

```py
#Options (could be optimised for preprocessing)
dropout = True
normalization = True
summary = True
```
To see the Tensorboard results : 
```sh
$ tensorboard --logdir ./logs_conv
```
To run the network : 
```sh
$ ./train_mnist_conv.py
```
You might need to make the file executable : 

```sh
$ chmod u+x train_mnist_conv.py
```
### Second approach to convolutional network with less parameters.

In the next commit.