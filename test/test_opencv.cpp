#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/surface_matching.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include "opencv2/surface_matching/ppf_match_3d.hpp"
#include <iostream>
#include <string>

/** function main **/
int main(int argc, char** argv)
{
/*	cv::Mat img_object = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	cv::namedWindow( "window" );
	cv::imshow("window",img_object);
	cv::waitKey(0);*/

	cv::Mat image;
	cv::ppf_match_3d::ICP icp;
	std::string filename = "../data/cube.ply";
	cv::ppf_match_3d::loadPLYSimple(filename.c_str(), 0).copyTo(image);
	cv::Mat m1;
	m1 = cv::Mat::zeros(4,4,CV_64F);
	std::cout << m1.at<double>(0,0) << "\n";

	std::cout << image << "\n";

  return 0;
}