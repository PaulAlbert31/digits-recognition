# Digits recognition

This is a personal project i work on during my free time. Feel free to try it yourself, i will add the instruction further down. I use the C++ language for now but i might switch to python in the future.

# What's new ?

  - I am currently working on a better way to represent digits that is scale-resistant as well a invariant in rotation. See https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/mori-cvpr01.pdf


Yet to be done:
  - Unsupervised classification using online k-means.
    - Improve the online k-means by using Kohonen maps.
    - Improve the representation of the digit to make it robust to scale and rotation
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
This will create a number of images in the current directory
To make a short film out of it :

```sh
$ avconv -i kmeans-%06d.ppm -b:v 1M kmeans.avi
```
To delete all images + film :
```sh
$ find . -name 'kmeans*' -delete
```



