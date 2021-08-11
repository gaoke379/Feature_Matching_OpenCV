#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"



int main(int argc, char** argv)
{
    cv::initModule_nonfree();
    std::string img1_path, img2_path;
    std::string detector_type, descriptor_type, matcher_type;
    if (argc != 6) {
        img1_path = "../../data/img1.ppm";
        img2_path = "../../data/img4.ppm";
        detector_type = "SIFT"; // FAST, ORB, SIFT, SURF, BRISK
        descriptor_type = "SIFT"; // ORB, SIFT, SURF, BRISK
        matcher_type = "DistRatio"; // DistRatio, NN
    }
    else {
        img1_path = argv[1];
        img2_path = argv[2];
        detector_type = argv[3];
        descriptor_type = argv[4];
        matcher_type = argv[5];
    }
    std::cout << "Query image: " << img1_path << std::endl;
    std::cout << "Match image: " << img2_path << std::endl;
    std::cout << "Feature detector: " << detector_type << std::endl;
    std::cout << "Feature descriptor: " << descriptor_type << std::endl;
    std::cout << "Matching scheme: " << matcher_type << std::endl;


    ///-- Initialization
    ///
    double scale_factor = 1.0; // scale factor for resizing the input images
    int thresh_fast = 69; // FAST feature detection threshold
    int num_ft_orb = 5000; // number of ORB features
    double num_ft_sift = 1000; // number of SIFT features
    int min_hess_surf = 670; // SURF feature detection threshold
    int thresh_brisk = 98;  // BRISK detection threshold score
    const float ratio_match = 0.8; // threshold ratio between best and second-best match; SIFT: 0.8; SURF: 0.7
    double nn_match_thresh = 90; // matching threshold for nearest neighbor


    ///-- Load input images
    ///
    cv::Mat img1 = cv::imread(img1_path, CV_LOAD_IMAGE_COLOR); //read the first image as the query image
    cv::Mat img2 = cv::imread(img2_path, CV_LOAD_IMAGE_COLOR); //read the second image as the match image
    if (!img1.data || !img2.data ) { // check whether the image is loaded or not
        std::cout << "Error : Images cannot be loaded..!!" << std::endl;
        return EXIT_FAILURE;
    }
    if (abs(scale_factor - 1.0) > 0.01) {
        cv::resize(img1, img1, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
        cv::resize(img2, img2, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
    }
    cv::Mat img1_gray, img2_gray;
    //-- Convert to grayscale image if RGB
    if (img1.channels() == 3) {
        cv::cvtColor(img1, img1_gray, CV_BGR2GRAY);
    }
    else {
        img1_gray = img1.clone();
    }
    if (img2.channels() == 3) {
        cv::cvtColor(img2, img2_gray, CV_BGR2GRAY);
    }
    else {
        img2_gray = img2.clone();
    }


    ///-- Feature detection
    ///
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    clock_t t_detect_start = std::clock();
    if (detector_type == "FAST") {
        cv::FASTX(img1_gray, keypoints1, thresh_fast, true, cv::FastFeatureDetector::TYPE_9_16);
        cv::FASTX(img2_gray, keypoints2, thresh_fast, true, cv::FastFeatureDetector::TYPE_9_16);
    }
    else if (detector_type == "ORB") {
        cv::OrbFeatureDetector Detector(num_ft_orb, 1.2f, 8, 31, 0);
        Detector.detect(img1_gray, keypoints1);
        Detector.detect(img2_gray, keypoints2);
    }
    else if (detector_type == "SIFT") {
        cv::SiftFeatureDetector Detector(num_ft_sift, 1.1);
        Detector.detect(img1_gray, keypoints1);
        Detector.detect(img2_gray, keypoints2);
    }
    else if (detector_type == "SURF") {
        cv::SurfFeatureDetector Detector(min_hess_surf);
        Detector.detect(img1_gray, keypoints1);
        Detector.detect(img2_gray, keypoints2);
    }
    else if (detector_type == "BRISK") {
        cv::BRISK BRISKD(thresh_brisk, 3, 1.0);
        BRISKD.detect(img1_gray, keypoints1);
        BRISKD.detect(img2_gray, keypoints2);
    }
    clock_t t_detect_elapsed = std::clock() - t_detect_start;
    std::cout << "Number of feature points detected in image1: " << keypoints1.size() << std::endl;
    std::cout << "Number of feature points detected in image2: " << keypoints2.size() << std::endl;
    std::cout << "Total elapsed time for feature detection: " << ((float)t_detect_elapsed)/CLOCKS_PER_SEC << " s" << std::endl;
    //-- draw keypoints and save images
    cv::Mat img_keypoints1, img_keypoints2;
    cv::drawKeypoints(img1, keypoints1, img_keypoints1, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DEFAULT);
    cv::drawKeypoints(img2, keypoints2, img_keypoints2, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DEFAULT);
    cv::imwrite("keypoints1.png", img_keypoints1);
    cv::imwrite("keypoints2.png", img_keypoints2);


    ///-- Generate feature descriptors
    ///
    cv:: Mat descriptors1, descriptors2;
    clock_t t_desc_start = std::clock();
    if (descriptor_type == "SIFT") {
        cv::SiftDescriptorExtractor Extractor;
        Extractor.compute(img1_gray, keypoints1, descriptors1);
        Extractor.compute(img2_gray, keypoints2, descriptors2);
    }
    else if (descriptor_type == "SURF") {
        cv::SurfDescriptorExtractor Extractor;
        Extractor.compute(img1_gray, keypoints1, descriptors1);
        Extractor.compute(img2_gray, keypoints2, descriptors2);
    }
    else if (descriptor_type == "BRISK") {
        cv::BRISK BRISKD(thresh_brisk, 3, 1.0);
        BRISKD.compute(img1_gray, keypoints1, descriptors1);
        BRISKD.compute(img2_gray, keypoints2, descriptors2);
    }
    else if (descriptor_type == "ORB") {
        cv::OrbDescriptorExtractor Extractor;
        Extractor.compute(img1_gray, keypoints1, descriptors1);
        Extractor.compute(img2_gray, keypoints2, descriptors2);
    }
    clock_t t_desc_elapsed = std::clock() - t_desc_start;
    std::cout << "Total elapsed time for generating feature descriptors: " << ((float)t_desc_elapsed)/CLOCKS_PER_SEC << " s" << std::endl;
    //-- make sure the descriptors to be type CV_32F (non-binary descriptors) or CV_8U (binary descriptors)
    if (descriptor_type == "SIFT" || descriptor_type == "SURF") {
        if(descriptors1.type() != CV_32F)
            descriptors1.convertTo(descriptors1, CV_32F);
        if(descriptors2.type() != CV_32F)
            descriptors2.convertTo(descriptors2, CV_32F);
    }
    else if (descriptor_type == "BRISK" || descriptor_type == "ORB") {
        if(descriptors1.type() != CV_8U)
            descriptors1.convertTo(descriptors1, CV_8U);
        if(descriptors2.type() != CV_8U)
            descriptors2.convertTo(descriptors2, CV_8U);
    }


    ///-- Descriptor matching
    ///
    std::vector<cv::DMatch> matches, matches_good;
    int norm_type = 1; // brute-force matcher norm type
    if (descriptor_type == "SIFT" || descriptor_type == "SURF") {
        norm_type = cv::NORM_L2;
    }
    else if (descriptor_type == "BRISK" || descriptor_type == "ORB") {
        norm_type = cv::NORM_HAMMING;
    }
    cv::BFMatcher bf_matcher(norm_type); // use NORM_HAMMING for BRISK and ORB
    if (matcher_type == "NN") {
        //-- nearest neighbor matching
        bf_matcher.match(descriptors1, descriptors2, matches); // find the best match
    }
    else if (matcher_type == "DistRatio") {
        //-- implement SIFT matching scheme ------------------------
        //-- brute-force matching first ----------------------------
        //-- compute ratio between best and second-best match ------
        std::vector<std::vector<cv::DMatch> > matches_all;
        bf_matcher.knnMatch(descriptors1, descriptors2, matches_all, 2); // find two nearest matches
        for (size_t j = 0; j < matches_all.size(); j++) {
            if (matches_all[j][0].distance < ratio_match * matches_all[j][1].distance) {
                matches.push_back(matches_all[j][0]);
            }
        }
    }
    for (int j = 0; j < matches.size(); j++) {
        if (matcher_type == "NN") {
            if (matches[j].distance <= nn_match_thresh) {
                matches_good.push_back(matches[j]);
            }
        }
        else if (matcher_type == "DistRatio") {
            matches_good.push_back(matches[j]);
        }
    }
    std::cout << "Number of good matches: " << matches_good.size() << std::endl;
    //-- Save a side-by-side image showing matches on it
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches_good, img_matches,
            cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    std::string saveName = "matches.png";
    cv::imwrite(saveName, img_matches);


    return EXIT_SUCCESS;
}
