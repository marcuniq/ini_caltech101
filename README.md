# Bachelor Practical
## Image Classification with Convolutional Neural Nets

The task for this bachelor practical was to train a convolutional neural network on the Caltech-101 dataset.

My learnings:
* improved knowledge of technologies and concepts (Python, Neural Networks, Machine Learning)
* got to know new technologies and concepts (Keras, Theano, Batch Normalization)

The full report for the bachelor practical can be found here: [report.pdf](/report/report.pdf)

## Getting the Code

To get a local copy of the code, clone it using git:

    git clone https://github.com/marcuniq/ini_caltech101.git
    cd ini_caltech101

Make sure you have the bleeding edge version of Theano, or run

    pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

Next, install the package. Use 'develop' instead of 'install' if you consider changing package code

    python setup.py develop
    
Run train.sh (sets proper theano env flags), which downloads and untars the 'img-gen-resized' dataset, then starts training.

    ./train.sh

## Example of the generated images in the 'img-gen-resized' dataset
![original](http://www.googledrive.com/host/0B6t56IB_eb6hbzlDX1RBeS00dW8)
![generated image 1](http://www.googledrive.com/host/0B6t56IB_eb6hSkFNVnFoT3Jlbkk)
![generated image 2](http://www.googledrive.com/host/0B6t56IB_eb6hejFXVVRTaGFYM2s)

[More examples](https://drive.google.com/folderview?id=0B6t56IB_eb6hVFRGOFp3QVpaR2M&usp=sharing)