# Capstone Project in course: CUDA advanced libraries

## Overview
This is an image processing program to run a 2D convolution conversion on an image. It utilizes the cuDNN (CUDA Deep Neural Network) libraries which enhance the efficiency by using CUDA and its parallel processing of GPUs.
More specifically, the convolution kernel/filters this program uses is the sober filter (edge detection). This programs allows user to specify an image file as input, do convolution progressing and output the output image with edge enhanced information into a new files.
As example, an input and output files are provided, and an video demo is attached with this project in the submission. 


### What is a 2D Convolution?
A 2D convolution is a mathematical operation where a small matrix (called a kernel or filter) slides over an input matrix (such as an image) to extract features. The operation consists of:

* Placing the kernel at a position on the input matrix.
* Multiplying the corresponding elements of the kernel and the input.
* Summing up the products to compute the output pixel value.
* Sliding the kernel to the next position and repeating.
* This process results in a transformed output matrix, capturing patterns like edges, textures, and shapes.

### More background on 2D convolution:

https://medium.com/@ml_dl_explained/understanding-2d-convolutions-in-pytorch-b35841149f5f

https://www.youtube.com/watch?v=pmyulQwV62k&t=3s


Sobel Filter:

https://en.wikipedia.org/wiki/Sobel_operator

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.


## Exaplanation and Tutorial

### How to build
```
make clean all
```

### How to run
```
./run.sh
or
make run
or 
bin/ImageConvolution.exe --input data/sloth.png
```

### Where is the input and output
#### Input
You can using --input in command line to specifiy the input image, or use the default example image as input by entering ```make run``` or ```run.sh```

#### Output
A file with suffix ```_processed``` in ```data/``` folder will be populated.


### Screen output of example run
```
$ ./run.sh 
rm -rf bin/*; rm -f data/*_processed.*
# mkdir -p bin
g++ -std=c++17  -I/usr/local/cuda/include -Iinclude -I/usr/include -I/usr/local/cuda/targets/x86_64-linux/include -I/usr/include/opencv4/opencv -I/usr/include/opencv4 src/ImageConvolution.cpp -o bin/ImageConvolution.exe -L/usr/local/cuda/lib64 -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lcuda -lcudnn -lopencv_core -lopencv_imgcodecs

$ bin/ImageConvolution.exe (or)
$ bin/ImageConvolution.exe --input data/sloth_backup.png
Loading input image file:data/sloth.png
Input image: 1666x1250 with 3 channels
Output dimensions: 1250x1666 with 3 channels
Selected algorithm: 1
Workspace needed: 0 bytes
Convolution completed!
Output image saved to: data/sloth_processed.png

$ bin/ImageConvolution.exe --input data/scene-Iron-Man.png
Loading input image file:data/scene-Iron-Man.png
Input image: 1600x1067 with 3 channels
Output dimensions: 1067x1600 with 3 channels
Selected algorithm: 1
Workspace needed: 0 bytes
Convolution completed!
Output image saved to: data/scene-Iron-Man_processed.png
```

