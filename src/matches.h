#ifndef matches_hpp
#define matches_hpp
#include <vector>
#include <opencv2/opencv.hpp>
#include "keypoints.h"
#include <fstream>

using namespace std;

class Matches {
private:
	void pairwiseMatch(const vector<cv::KeyPoint>&keypoints1, const vector<cv::KeyPoint>&keypoints2, const cv::Mat&descriptors1, const cv::Mat&descriptors2, vector<cv::DMatch>& matches);
	vector<vector<vector<cv::DMatch> > > matchesTable;
public:
	Matches(const KeyPoints&keyPoints);
	Matches(const string&fileName);
	const vector<cv::DMatch>& getMatches(int i, int j) const;
	void saveTo(const string&fileName);
};
#endif
