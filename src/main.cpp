#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
// #include <cmath>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



int main(int argc, char** argv)
{
    std::string img1_path, img2_path;
    std::string feature_name, matcher_type;
    if (argc != 5) {
        img1_path = "../../data/img1.ppm";
        img2_path = "../../data/img4.ppm";
        feature_name = "SIFT"; // SIFT, SURF, AKAZE, ORB
        matcher_type = "DistRatio"; // SIFT/SURF/AKAZE: DistRatio
    }
    else {
        img1_path = argv[1];
        img2_path = argv[2];
        feature_name = argv[3];
        matcher_type = argv[4];
    }
    bool flag_output_fts_img = true;
    bool flag_output_matches_img = true;
    bool flag_output_fts_txt = false;
    bool flag_output_matches_txt = false;
    std::cout << "Reference image: " << img1_path << std::endl;
    std::cout << "Matching image: " << img2_path << std::endl;
    std::cout << "Feature: " << feature_name << std::endl;
    std::cout << "Matching scheme: " << matcher_type << std::endl;


    ///-- Initialization
    ///
    double scale_factor = 1.0; // scale factor for resizing the input images
    int num_ft_sift = 1000; // number of SIFT features
    int min_hess_surf = 800; // SURF feature detection threshold
    float thresh_akaze = 0.0015f; // AKAZE feature detection threshold
    int num_ft_orb = 1000; // number of ORB features
    double ratio_match = 0.8; // threshold ratio between best and second-best match; SIFT/AKAZE: 0.8; SURF: 0.7
    double nn_match_thresh = 60; // matching threshold for nearest neighbor


    ///-- Load input images
    ///
    cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_COLOR); // read the first image as the query image
    cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_COLOR); // read the second image as the match image
    //-- Check if the images are loaded
    if (!img1.data || !img2.data) {
        std::cout << "Error : Images cannot be loaded!" << std::endl;
        return EXIT_FAILURE;
    }
    //-- Resize images if scale factor is not 1.0
    if (std::abs(scale_factor - 1.0) > 0.01) {
        cv::resize(img1, img1, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
        cv::resize(img2, img2, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
    }
    //-- Convert to grayscale image if RGB
    cv::Mat img1_gray, img2_gray;
    if (img1.channels() == 3) {
        cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    }
    else {
        img1_gray = img1.clone();
    }
    if (img2.channels() == 3) {
        cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    }
    else {
        img2_gray = img2.clone();
    }


    ///-- Feature extraction (detection + description)
    ///
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv:: Mat descriptors1, descriptors2;

    if (feature_name == "SIFT") {
        cv::Ptr<cv::Feature2D> sift1 = cv::xfeatures2d::SIFT::create(num_ft_sift);
        sift1->detectAndCompute(img1_gray, cv::Mat(), keypoints1, descriptors1);
        cv::Ptr<cv::Feature2D> sift2 = cv::xfeatures2d::SIFT::create(num_ft_sift);
        sift2->detectAndCompute(img2_gray, cv::Mat(), keypoints2, descriptors2);
    }
    else if (feature_name == "SURF") {
        cv::Ptr<cv::Feature2D> surf1 = cv::xfeatures2d::SURF::create(min_hess_surf);
        surf1->detectAndCompute(img1_gray, cv::Mat(), keypoints1, descriptors1);
        cv::Ptr<cv::Feature2D> surf2 = cv::xfeatures2d::SURF::create(min_hess_surf);
        surf2->detectAndCompute(img2_gray, cv::Mat(), keypoints2, descriptors2);
    }
    else if (feature_name == "AKAZE") {
        cv::Ptr<cv::AKAZE> akaze1 = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, thresh_akaze);
        akaze1->detectAndCompute(img1_gray, cv::noArray(), keypoints1, descriptors1);
        cv::Ptr<cv::AKAZE> akaze2 = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, thresh_akaze);
        akaze2->detectAndCompute(img2_gray, cv::noArray(), keypoints2, descriptors2);
    }
    else if (feature_name == "ORB") {
        cv::Ptr<cv::ORB> orb1 = cv::ORB::create(num_ft_orb);
        orb1->detectAndCompute(img1_gray, cv::Mat(), keypoints1, descriptors1);
        cv::Ptr<cv::ORB> orb2 = cv::ORB::create(num_ft_orb);
        orb2->detectAndCompute(img2_gray, cv::Mat(), keypoints2, descriptors2);
    }
    else {
        std::cout << "Invalid feature..." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Number of feature points detected in image1: " << keypoints1.size() << std::endl;
    std::cout << "Number of feature points detected in image2: " << keypoints2.size() << std::endl;
    //-- Draw keypoints and save images
    if (flag_output_fts_img) {
        cv::Mat img_keypoints1, img_keypoints2;
        cv::drawKeypoints(img1, keypoints1, img_keypoints1, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DEFAULT);
        cv::drawKeypoints(img2, keypoints2, img_keypoints2, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DEFAULT);
        cv::imwrite("img1_ft_" + feature_name + ".jpg", img_keypoints1);
        cv::imwrite("img2_ft_" + feature_name + ".jpg", img_keypoints2);
    }
    //-- Export feature points to a txt file
    FILE *datafile_ft1, *datafile_ft2;
    std::string save_name1 = "img1_ft.txt";
    std::string save_name2 = "img2_ft.txt";
    if (flag_output_fts_txt) {
        datafile_ft1 = fopen(save_name1.c_str(), "w");
        datafile_ft2 = fopen(save_name2.c_str(), "w");
        for (size_t i = 0; i < keypoints1.size(); i++) {
            fprintf(datafile_ft1, "%.6f %.6f\n", keypoints1[i].pt.x, keypoints1[i].pt.y); // write to txt file
        }
        for (size_t i = 0; i < keypoints2.size(); i++) {
            fprintf(datafile_ft2, "%.6f %.6f\n", keypoints2[i].pt.x, keypoints2[i].pt.y); // write to txt file
        }
        fclose(datafile_ft1);
        fclose(datafile_ft2);
    }

    ///-- Descriptor matching
    ///
    std::vector<cv::DMatch> matches, matches_good;
    int norm_type = 1; // brute-force matcher norm type
    if (feature_name == "SIFT" || feature_name == "SURF") {
        norm_type = cv::NORM_L2;
    }
    else if (feature_name == "AKAZE" || feature_name == "ORB") {
        norm_type = cv::NORM_HAMMING;
    }
    cv::BFMatcher bf_matcher(norm_type);
    if (matcher_type == "NN") {
        //-- nearest neighbor matching
        bf_matcher.match(descriptors1, descriptors2, matches); // find the best match
    }
    else if (matcher_type == "DistRatio") {
        //-- distance ratio matching
        std::vector<std::vector<cv::DMatch> > matches_all;
        bf_matcher.knnMatch(descriptors1, descriptors2, matches_all, 2); // find two nearest matches
        for (size_t j = 0; j < matches_all.size(); j++) {
            if (matches_all[j][0].distance < ratio_match * matches_all[j][1].distance) {
                matches_good.push_back(matches_all[j][0]);
            }
        }
    }
    //-- Select good matches from nearest neighbor matches
    if (matcher_type == "NN") {
        for (int j = 0; j < matches.size(); j++) {
            if (matches[j].distance <= nn_match_thresh) {
                matches_good.push_back(matches[j]);
            }
        }
    }
    std::cout << "Number of good matches: " << matches_good.size() << std::endl;
    //-- Save a side-by-side image showing matches on it
    if (flag_output_matches_img) {
        cv::Mat img_matches;
        cv::drawMatches(img1, keypoints1, img2, keypoints2, matches_good, img_matches,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        std::string saveName = "img1_img2_" + feature_name + "_matches.jpg";
        cv::imwrite(saveName, img_matches);
    }
    //-- Export feature matches to a txt file
    std::string save_name_match = "img1_img2_" + feature_name + "_matches.txt";
    FILE *datafile_match;
    if (flag_output_matches_txt) {
        datafile_match = fopen(save_name_match.c_str(), "w");
        for (size_t i = 0; i < matches_good.size(); i++) {
            cv::KeyPoint kpt1 = keypoints1[matches_good[i].queryIdx];
            cv::KeyPoint kpt2 = keypoints2[matches_good[i].trainIdx];
            fprintf(datafile_match, "%.6f %.6f %.6f %.6f\n", kpt1.pt.x, kpt1.pt.y, kpt2.pt.x, kpt2.pt.y); // write to txt file
        }
        fclose(datafile_match);
    }


    return EXIT_SUCCESS;
}
