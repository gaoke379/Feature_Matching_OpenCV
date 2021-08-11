# Feature Matching Using OpenCV

### Description
C++ code for running feature extraction and matching on two input images using the following features in OpenCV:  
- `SIFT`  
- `SURF`  
- `AKAZE`  
- `ORB`    


### Build Library Dependencies  
Several dependencies need to be available on the system:  
- `OpenCV C++: 3.0.0 ~ 3.4.10 or 4.0.0 ~ 4.3.0 `
- `CMake >= 2.6`  

Note that SIFT has been moved to the main OpenCV repository (patent on SIFT is expired) starting from 3.4.11 and 4.4.0, so the function call to SIFT has to be changed from `cv::xfeatures2d::SIFT` to `cv::SIFT` if newer versions of OpenCV is being used.  


### Configure and Build  
```
$ cd Feature_Matching_OpenCV
$ mkdir build  
$ cd build    
$ cmake ..    
$ make   
```


### Running the code     
To run the feature matching code, go to `bin/linux` directory and invoke `./FeatureMatch <image1_path> <image2_path> <feature_name> <matcher_type>`      
Example usage: `./FeatureMatch ../../data/img1.ppm ../../data/img4.ppm SIFT DistRatio`  


### Releases and Changes  
- **(Branch) master**: feature detection and matching code for OpenCV version 3 & 4.   
- **(Branch) opencv2**: includes code changes according to the OpenCV version 2.x.x.
