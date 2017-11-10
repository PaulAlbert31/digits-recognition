# Digits recognition

This is a personal project i work on during my free time. Feel free to try it yourself, i will add the instruction further down. I use the C++ language for now but i might switch to python in the future.
The document Base.h is a courtesy of Hervé Frezza-Buet : http://www.metz.supelec.fr/metz/personnel/frezza/

# What's new ?

  - I am currently working on a better way to represent digits that is scale-resistant as well a invariant in rotation. See https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/mori-cvpr01.pdf


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
- Deep learning methods

# Want to run it yourself ?

### Kmeans + Kokonen maps
```sh
$ cd digits-recognition
$ g++ -o online-kmeans-kohonen -Wall -ansi -O3 online-kmeans-kohonen.cc
$ ./online-kmeans-kohonen
```
This will create a number of images in the current directory, please set VERBOSE to false if you don't want to.
```sh
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
```sh
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
```sh
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