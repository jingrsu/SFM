#ifndef keypoints_hpp
#define keypoints_hpp
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

class KeyPoints {
private:
	vector<vector<cv::KeyPoint> > keypointsVec;
	vector<cv::Mat> descriptorsVec;
public:
	KeyPoints(const vector<cv::Mat>&images);
	KeyPoints(const string&fileName);
	int getFrameNum() const;
	int getKeyPointNum(int i) const;
	cv::KeyPoint getKeyPoint(int i, int j) const;
	const vector<cv::KeyPoint>& getKeyPoints(int i) const;
	const cv::Mat& getDescriptors(int i) const;
	void saveTo(string fileName);
};
#endif