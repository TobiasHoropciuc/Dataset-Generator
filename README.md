# Dataset-Generator

A small python project offering image dataset generators for classification and detection tasks.

This project is part of my diploma thesis and was the developed to decrease data collection needs and increase productivity of deep learning development for image classification and object detection. The two main components are the generators defined in `generator.py`. They enable users to augment datasets through image manipulations/filters and combining object images with backgrounds, which is detection specific. This software implements various augmentation methods like cropping, color jittering and random erase. Furthermore, an annotation writer was implemented for generating object detetion datasets, which can generate  XML and JSON annotation files. 

## Setup

In order to use these generators one must first install all the necessary python libraries. The following packages must be installed via pip:

- NumPy: `pip install numpy`
- OpenCV: `pip install cv2`
- Albumentations: `pip install albumentations`
- Pascal Voc Writer: `pip install pascal-voc-writer`

It is recommended to install the above mentioned packages in a virtual environment. To do so, create the venv inside the root-project-directory and active it before installing the Python packages.

```bash
$ python -m venv venv 
$ source /venv/bin/activate
```

## Running the Generators

The project offers two types of generators, one for image-classification and one for object-detection. Both are connected through `generate.py` and can be run by executing this script. To configure which generator should be used, a INI-config-file must be passed to the script. The configuration file must include all the necessary parameters for the generator as well as the augmentation policy. Examples for valid configurations for generators and all available  augmentations are given in `config/classification.ini` and `config/detection.ini`. After setting up the configuration, the project can be run as follows:

```bash
$ python generate.py --config /path/to/config/file 
```

For easy experimentation you can execute the program the included config files and images. Keep in mind that you always have to active the venv before running the project.
