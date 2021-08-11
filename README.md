# Feature Matching Using OpenCV

### Description
C++ code for running feature extraction and matching on two input images using the following features in OpenCV version 2.x.x:  
- `SIFT`  
- `SURF`  
- `BRISK`  
- `ORB`    


### Build Library Dependencies  
Several dependencies need to be available on the system:  
- `OpenCV C++: 2.x.x`
- `CMake >= 2.6`  



### Configure and Build  
```
$ cd Feature_Matching_OpenCV
$ mkdir build  
$ cd build    
$ cmake ..    
$ make   
```


### Running the code     
To run the feature matching code, go to `bin/linux` directory and invoke `./FeatureMatch <image1_path> <image2_path> <detector_type> <descriptor_type> <matcher_type>`      
Example usage: `./FeatureMatch ../../data/img1.ppm ../../data/img4.ppm SIFT SIFT DistRatio`  

