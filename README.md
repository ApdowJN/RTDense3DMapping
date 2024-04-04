# Real-Time Dense 3D Mapping of Underwater Environments
Real-Time Dense 3D Mapping repository. This pipeline address the problem of real-time dense reconstruction for a resource constrained autonomous underwater robot.

# Prerequisites
We have tested the library in **16.04** and **18.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++11 or C++0x Compiler
We use the STL and chrono functionalities of C++11.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **Tested with  OpenCV 3.3.1 and OpenCV 4.5.5**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## Other dependencies
```bash
sudo apt install libceres-dev libgflags-dev
```

# Building our pipeline and example
The code has been tested on Ubuntu 16.04 and Ubuntu 18.04, with gcc 7.5.

Clone the repository:

```
git clone https://github.com/ApdowJN/RTDense3DMapping.git
```

To build our pipeline:

```
cd IROS/
mkdir build/
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j $(nproc)
```

# How To Run Example
## Mexico Dataset

1. Download the <a href="https://stevens0-my.sharepoint.com/:f:/g/personal/nburgdor_stevens_edu/EvzIpSlbG-RCpOcLCrQSqJcBNvlTyyHKcNIcOK67HodecQ?e=KC8iS9">Mexico<a> dataset.

2. Extract images.zip to obtain the `left/` and `right/` image folders

The folder structure should mirror the following:

```
Mexico
-> left
    -> #######.png
    -> #######.png
-> right
    -> #######.png
    -> #######.png
-> colmap.ply
-> filename_mapping.csv
-> intrinsics.yaml
-> poses.txt
```

3. Execute the following command to run example in Mexico dataset
``` bash
cd scipts
./run_mexico.sh
```

The path paramters inside the script may need to be changed to reflect where the data has been stored. To do so, open the script and modify the first two variables:

``` bash
DATA_PATH=<path-to-dataset>
OUTPUT_PATH=<desired-output-path>
```

This example extends to all the other datasets.

# Datasets
Here you can find links to additional datasets:
-   <a href="https://stevens0-my.sharepoint.com/:f:/g/personal/nburgdor_stevens_edu/EvgvYaDF4qdPoRASKn2B318BlV8T-7_VyEqL_yPkxDb4DQ?e=9XhvSy">Florida<a>
-   <a href="https://stevens0-my.sharepoint.com/:f:/g/personal/nburgdor_stevens_edu/EgFT_5jsyN1EuOH7sUE_lGQB0hxmUZpoDOtk_rAFDxZbpg?e=nD9RBk">Pamir<a>
-   <a href="https://stevens0-my.sharepoint.com/:f:/g/personal/nburgdor_stevens_edu/EkHQhpQRpBhKmQN_l1uWgygBVM4WKuU3GjzC64wgCVkNVw?e=tcglAi">Stavronikita<a>
    
# Data File Formats
## Filename Mapping
The filename mapping must follow the format:

```
<timestamp>,<timestamp>.png
<timestamp>,<timestamp>.png
<timestamp>,<timestamp>.png
<timestamp>,<timestamp>.png
.
.
.
```

where the `<timestamp>` matches the timestamps in the `poses.txt` file.
        
## Intrinsics
The intrinsics file must adhere to the following format:

```
%YAML:1.0

left: 
K: !!opencv-matrix
    rows: 3
    cols: 3
    dt: d
    data: [574.760004965, 0.0, 399.37765462, 0.0, 575.289618425,  288.60124274, 0.0, 0.0, 1.0]
D: !!opencv-matrix
    rows: 1
    cols: 4
    dt: d
    data: [0.0, 0.0, 0.0, 0.0]

right:
K: !!opencv-matrix
    rows: 3
    cols: 3
    dt: d
    data: [575.6337256, 0.0, 391.22448089, 0.0, 574.43588977, 286.86941794, 0.0, 0.0, 1.0]
D: !!opencv-matrix
    rows: 1
    cols: 4
    dt: d
    data: [0.0, 0.0, 0.0, 0.0]

# image resolution [ width x height ]
size: [800, 600]

# baseline (mm)
baseline: 142.20

# rotation matrix to as input to stereo rectify
R: !!opencv-matrix
    rows: 3
    cols: 3
    dt: d
    data: [0.9999531274148135, -0.0024947708286728633, 0.009355163912669094, 0.0022038626034325114, 0.99951762419007, 0.030978410595931,                    -0.009427935242973135, -0.030956341061845496, 0.9994762723472247]

# translation matrix to as input to stereo rectify
T: !!opencv-matrix
    rows: 1
    cols: 3
    dt: d
    data: [-0.1422002, 0.00018052, -0.00134952]
```
        
## Poses
The poses file must adhere to the following format:

```
<timestamp> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
<timestamp> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
<timestamp> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
<timestamp> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
<timestamp> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
.
.
.
```
            
If any of the formats change, the changes must be propagated to the data loading function in the `src/util.cpp` file.
