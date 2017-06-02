#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    cv::initModule_nonfree();
    const char* query_imaga_filename = "/home/kegao/Desktop/Research/Dataset/TS_Albq215_Crop01/BBox_001_001.png";
    const char* train_imaga_filename = "/home/kegao/Desktop/Research/Dataset/TS_Albq215_Crop01/BBox_001_007.png";
    Mat image1 = imread(query_imaga_filename, CV_LOAD_IMAGE_COLOR); //read the first image as the query image
    Mat image2 = imread(train_imaga_filename, CV_LOAD_IMAGE_COLOR); //read the first image as the train image

    if (!image1.data || !image2.data ) //check whether the image is loaded or not
    {
        cout << "Error : Images cannot be loaded..!!" << endl;
        return -1;
    }

    Mat imgGray1, imgGray2;
    if (image1.channels() == 3) {
        cv::cvtColor(image1, imgGray1, CV_BGR2GRAY); }
    else {
        imgGray1 = image1.clone(); }
    if (image2.channels() == 3) {
        cv::cvtColor(image2, imgGray2, CV_BGR2GRAY); }
    else {
        imgGray2 = image2.clone(); }
    //namedWindow("Grayscale Image", CV_WINDOW_AUTOSIZE); //create a window named "Grayscale Image"
    //imshow("Grayscale Image", gray);  //display the image in the window created

    vector<KeyPoint> keyp1, keyp2;

    clock_t tDetectStart = clock();

    //------------ FAST -----------------------------------
//    int threshold = 80; //threshold on difference between intensity of the central pixel and pixels of a circle around this pixel
//    FASTX(imgGray1, keyp1, threshold, true, FastFeatureDetector::TYPE_9_16 );
//    FASTX(imgGray2, keyp2, threshold, true, FastFeatureDetector::TYPE_9_16 );
    //------------ ORB -----------------------------------
//    OrbFeatureDetector Detector (200, 1.2f, 8, 31, 0);
//    Detector.detect(imgGray1, keyp1);
//    Detector.detect(imgGray2, keyp2);
    //------------ SIFT -------------------------------------
    double threshSIFT = 0.9, edgeThreshSIFT = 1.0;
    SiftFeatureDetector Detector (threshSIFT, edgeThreshSIFT);
    Detector.detect(imgGray1, keyp1);
    Detector.detect(imgGray2, keyp2);
    //------------ SURF ------------------------------------
//    int minHess = 3500;
//    SurfFeatureDetector Detector (minHess);
//    Detector.detect(imgGray1, keyp1);
//    Detector.detect(imgGray2, keyp2);
    //------------ BRISK ----------------------------------
//    int threshBRISK = 30; // FAST/AGAST detection threshold score
//    int octavesBRISK = 3; // detection octaves. Use 0 to do single scale
//    float pScaleBRISK = 1.0; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint
//    BRISK BRISKD(threshBRISK, octavesBRISK, pScaleBRISK);
//    BRISKD.detect(imgGray1, keyp1);
//    BRISKD.detect(imgGray2, keyp2);


    clock_t tDetectElapsed = clock() - tDetectStart;
    cout<<"KeyPoint Size of Image1: "<<keyp1.size()<<endl;
    cout<<"KeyPoint Size of Image2: "<<keyp2.size()<<endl;
    cout << "Total elapsed time for feature detection: " << ((float)tDetectElapsed)/CLOCKS_PER_SEC << " s" << endl;

    //-- Draw keypoints
    Mat imgKeyp1, imgKeyp2;
    drawKeypoints( image1, keyp1, imgKeyp1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    drawKeypoints( image2, keyp2, imgKeyp2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imwrite("imgKeyp1.png", imgKeyp1);
    imwrite("imgKeyp2.png", imgKeyp2);

    //-- Feature matching
    float threshMatch = 150;
    Mat descriptor1, descriptor2;
    //------------- BRISK ---------------------
//    BRISKD.compute( imgGray1, keyp1, descriptor1 );
//    BRISKD.compute( imgGray2, keyp2, descriptor2 );
    //------------- SIFT -----------------------
    SiftDescriptorExtractor Extractor;
    Extractor.compute( imgGray1, keyp1, descriptor1 );
    Extractor.compute( imgGray2, keyp2, descriptor2 );
    //------------- SURF -----------------------
//    SurfDescriptorExtractor Extractor;
//    Extractor.compute( imgGray1, keyp1, descriptor1 );
//    Extractor.compute( imgGray2, keyp2, descriptor2 );

    if(descriptor1.type()!=CV_32F)
        descriptor1.convertTo(descriptor1, CV_32F);
    if(descriptor2.type()!=CV_32F)
        descriptor2.convertTo(descriptor2, CV_32F);
    FlannBasedMatcher Matcher;
    vector<DMatch> matches, matchesGood;
    Matcher.match( descriptor1, descriptor2, matches );
    for (int i1 = 0; i1 < matches.size(); i1++) {
        if (matches[i1].distance <= threshMatch) {
            matchesGood.push_back(matches[i1]);
        }
    }
    cout << "Number of matches: " << matches.size() << endl;
    cout << "Number of good matches: " << matchesGood.size() << endl;

    //-- Show matches
    Mat imgMatches;
    drawMatches( image1, keyp1, image2, keyp2, matchesGood, imgMatches,
            Scalar::all(-1), Scalar::all(-1),vector<char>(),
            DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    string saveName = "Good_Matches.png";
    //                imshow("good_matches",img_matches);
    imwrite(saveName, imgMatches);
    

    waitKey(0);
    return 0;
}



