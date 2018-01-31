#ifndef ImageSet_hpp
#define ImageSet_hpp
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

class ImageSet {
public:
	ImageSet(const string& imageDir, const string& parFileName);
	ImageSet(const string& imageDir);
	vector<cv::Mat> images;
	vector<string> imageNames;
	vector<cv::Mat> Kmats;
	vector<cv::Mat> RTmats;
	vector<cv::Mat> Pmats;

};
#endif