#include <iostream>
#include <vector>

#include "image.hpp"
#include "visualization.hpp"
#include <opencv2/opencv.hpp>

void display_H_graph(cv::Mat I_i, cv::Mat I_ip1, std::vector<cv::KeyPoint> kps1, std::vector<cv::KeyPoint> kps2, std::vector<cv::DMatch> matches, std::vector<int> inliers)
{
    cv::Mat imgMatches;
    cv::drawMatches(I_i,kps1,I_ip1,kps2,matches,imgMatches);
    
    for (int i=0; i<inliers.size(); ++i)
    {
        cv::line(imgMatches,kps1[inliers[i]].pt, kps2[inliers[i]].pt + cv::Point2f(I_i.cols, 0), cv::Scalar(0, 255, 0), 2);
    }

    // Show the result
    cv::imshow("Matches and Homography", imgMatches);
    cv::waitKey(0);
}

int main() 
{
    int ransac_n_iter = 500;
    float ransac_thr = 5.0;

    cv::Mat K = (cv::Mat_<float>(3, 3) << 320, 0, 480, 0, 320, 270, 0, 0, 1);

    // Read all images

    std::string path = "../images/";
    std::vector<cv::Mat> im_list;
    for (int i = 1; i < 9; ++i) {
        std::string im_file = path + std::to_string(i) + ".jpg";
        cv::Mat im = cv::imread(im_file);
        im_list.push_back(im);
    }

    // std::cout<<im_list.size()<<std::endl;
    // cv::imshow("Picture",im_list[0]);
    // cv::waitKey(0);

    // cv::destroyAllWindows();

    Image im;

    std::vector<cv::Mat> rot_list;
    std::vector<cv::Mat> H_list ;

    rot_list.push_back(cv::Mat::eye(3, 3, CV_32F));
    // for (size_t i = 0; i < 2; i++) {
    // Load consecutive images I_i and I_{i+1}
        cv::Mat I_i = im_list[0];
        cv::Mat I_ip1 = im_list[1];

        std::vector<cv::KeyPoint> kps1;
        std::vector<cv::KeyPoint> kps2;

        std::vector<cv::DMatch> matches;
        std::vector<cv::Point2f> x1, x2;
        std::vector<int> inliers;


        im.siftMatcher(I_i,I_ip1,matches,x1,x2,kps1,kps2);

        // std::cout<<kps1.size()<<"\t"<<kps2.size()<<std::endl;

        // Calculating Homography matrix H using Ransac
        H_list.push_back(im.estimateH(x1,x2,ransac_n_iter,ransac_thr,inliers));
    
        // Display homography connection between 2 images
        // display_H_graph(I_i,I_ip1,kps1,kps2,matches,inliers);

        rot_list.push_back(im.estimateR(H_list[0],K));


    int width = im_list[0].cols;
    int height = im_list[0].rows;

    int Hc = height;
    int Wc = im_list.size() * width/2;

    cv::Mat canvas = cv::Mat::zeros(Hc, Wc, CV_8UC3);
    cv::Mat p = im.constructCylindricalCoord(Wc, Hc, K);

    cv::Mat mask;

    cv::Mat u;
    u = im.projection(p, K,rot_list[0],Wc,Hc,mask);
    cv::Mat canvas_i = im.warpImage2Canvas(I_i,u,mask);
    
    

        
    // }

    /* 
    // Visualize the final canvas
    visualizeCanvas(canvas); 
    */

    return 0;
}
