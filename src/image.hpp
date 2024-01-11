#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <cmath>

class Image {
public:
    Image();
    ~Image();

    cv::Mat img1, img2;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // void extractSIFTFeatures();

    void siftMatcher(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& matchedPoints1,std::vector<cv::Point2f>& matchedPoints2, std::vector<cv::KeyPoint>& kps1, std::vector<cv::KeyPoint>& kps2);
    void findMatchesWithRatioTest(const cv::Mat& desc, const cv::Mat& descriptorsTarget, std::vector<cv::DMatch>& matches);
    cv::Mat estimateH(const std::vector<cv::Point2f>& x1, const std::vector<cv::Point2f>& x2, int ransac_n_iter, float ransac_thr, std::vector<int>& inliers);
    cv::Mat estimateR(const cv::Mat& H, const cv::Mat& K);
    cv::Mat constructCylindricalCoord(int Wc, int Hc, const cv::Mat& K);
    std::pair<cv::Mat, cv::Mat> meshgrid(const cv::Range& x_range, const cv::Range& y_range);
    cv::Mat projection(const cv::Mat& p, const cv::Mat& K, const cv::Mat& R, int W, int H, cv::Mat& mask);
    cv::Mat warpImage2Canvas(const cv::Mat& image_i, const cv::Mat& u, const cv::Mat& mask_i);
    

};

#endif