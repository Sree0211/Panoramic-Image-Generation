#include "image.hpp"

Image::Image() {}

Image::~Image() {}


void Image::siftMatcher(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& matchedPoints1,std::vector<cv::Point2f>& matchedPoints2, std::vector<cv::KeyPoint>& kps1, std::vector<cv::KeyPoint>& kps2) 
{
    // SIFT feature matching implementation
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();
    

    detector->detect(img1, kps1);
    detector->detect(img2, kps2);

    cv::Mat descriptors1, descriptors2;

    detector->compute(img1, kps1, descriptors1);
    detector->compute(img2, kps2, descriptors2);

    // Create matches to match keypoint descriptors
    findMatchesWithRatioTest(descriptors1, descriptors2, matches);

    // Extract keypoint locations from matches
    // std::vector<cv::Point2f> matchedPoints1, matchedPoints2;
    for (const auto& match : matches) {
        matchedPoints1.push_back(kps1[match.queryIdx].pt);
        matchedPoints2.push_back(kps2[match.trainIdx].pt);
    }

}

void Image::findMatchesWithRatioTest(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& matches)
{
    // Use a descriptor matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

    // Find matches using knn
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(desc1, desc2, knnMatches, 2);

    // Apply ratio test to filter out unnecessary matches.
    float ratioThreshold = 0.7;
    for (size_t i = 0; i < knnMatches.size(); ++i)
    {
        if (knnMatches[i].size() == 2 && knnMatches[i][0].distance < ratioThreshold * knnMatches[i][1].distance)
        {
            matches.push_back(knnMatches[i][0]);
        }
    }
    // std::cout<<matches.size()<<"\t"<<"kps2.size()"<<std::endl;
}

cv::Mat Image::estimateH(const std::vector<cv::Point2f>& x1, const std::vector<cv::Point2f>& x2, int ransac_n_iter, float ransac_thr, std::vector<int>& inliers) {
    /*  
    Homography estimation using RANSAC

    Idea is to map x1 to x2 (x2 = Hx1)
    
    */
    cv::Mat H = cv::findHomography(x1,x2,cv::RANSAC,ransac_thr,inliers,ransac_n_iter,0.9949);
    
    return H;
}



cv::Mat Image::estimateR(const cv::Mat& H, const cv::Mat& K) 
{
    // Compute the relative rotation matrix
    cv::Mat R;
    R = K.inv() * H;

    return R;

}

cv::Mat Image::constructCylindricalCoord(int Wc, int Hc, const cv::Mat& K) {
    // Generate 3D points on the cylindrical surface
    auto [canvasCoordsW, canvasCoordsH] = meshgrid(cv::Range(0, Wc), cv::Range(0, Hc));

    cv::Mat p(Hc, Wc, CV_32FC3);

    // Convert canvas coordinates to cylindrical coordinates
    float f = K.at<float>(0, 0); // K[1,1] is the focal length

    for (int row = 0; row < Hc; ++row) {
        for (int col = 0; col < Wc; ++col) {
            float phi = 2 * CV_PI * canvasCoordsW.at<float>(row, col) / Wc;
            float h = canvasCoordsH.at<float>(row, col);

            // Map cylindrical coordinates to 3D points
            p.at<cv::Vec3f>(row, col)[0] = f * std::cos(phi);
            p.at<cv::Vec3f>(row, col)[1] = h;
            p.at<cv::Vec3f>(row, col)[2] = f * std::sin(phi);
        }
    }

    return p;
}


std::pair<cv::Mat, cv::Mat> Image::meshgrid(const cv::Range& x_range, const cv::Range& y_range) {
    int x_size = x_range.end - x_range.start;
    int y_size = y_range.end - y_range.start;

    cv::Mat x_grid(y_size, x_size, CV_32F);
    cv::Mat y_grid(y_size, x_size, CV_32F);

    for (int i = 0; i < y_size; ++i) {
        x_grid.row(i).setTo(cv::Scalar(x_range.start), cv::noArray());
    }

    for (int i = 0; i < x_size; ++i) {
        y_grid.col(i).setTo(cv::Scalar(y_range.start), cv::noArray());
    }

    for (int i = 0; i < x_size; ++i) {
        x_grid.col(i) += i;
    }

    for (int i = 0; i < y_size; ++i) {
        y_grid.row(i) += i;
    }

    return std::make_pair(x_grid, y_grid);
}

cv::Mat Image::projection(const cv::Mat& p, const cv::Mat& K, const cv::Mat& R, int W, int H, cv::Mat& mask) 
{
    cv::Mat u(H, W, CV_32FC2);
    // cv::Mat mask(H, W, CV_8UC1, cv::Scalar(0));

    // Projection
    cv::Mat projectedPoints = K * R * p.t();
    cv::divide(projectedPoints.row(0), projectedPoints.row(2), u.col(0));
    cv::divide(projectedPoints.row(1), projectedPoints.row(2), u.col(1));

    // Create binary mask for valid pixels
    cv::inRange(u.col(0), 0, W - 1, mask);
    cv::inRange(u.col(1), 0, H - 1, mask);
    // std::make_pair(u, mask)
    return u;
}


cv::Mat Image::warpImage2Canvas(const cv::Mat& image_i, const cv::Mat& u, const cv::Mat& mask_i)
{
    cv::Mat canvas_i(image_i.size(), image_i.type(), cv::Scalar(0));

    // Copy pixel values from source image to canvas based on mapped locations
    for (int h = 0; h < canvas_i.rows; ++h) {
        for (int w = 0; w < canvas_i.cols; ++w) {
            if (mask_i.at<uchar>(h, w) > 0) {
                int u_warp = static_cast<int>(u.at<cv::Vec2f>(h, w)[0]);
                int v_warp = static_cast<int>(u.at<cv::Vec2f>(h, w)[1]);

                canvas_i.at<cv::Vec3b>(h, w) = image_i.at<cv::Vec3b>(v_warp, u_warp);
            }
        }
    }

    return canvas_i;
}
