#ifndef SFM_H
#define SFM_H
#include <opencv2/core/core.hpp>
#include "imageset.h"
#include "keypoints.h"
#include "matches.h"
#include "tracklist.h"

using namespace std;

class SFM
{
public:
	SFM(ImageSet& _imageset, KeyPoints& _keypoints, Matches& _matches, TrackList& _tracklist);
	void findBaselineTriangulation();
private:

	ImageSet & imageset;
	KeyPoints& keypoints;
	Matches& matches;
	TrackList& tracklist;

	void sortPairsForBaseline(vector<pair<float, pair<int, int>>>& pairs);
	void getMatchedPoints(int i, int j, vector<cv::Point2f>&points_i, vector<cv::Point2f>&points_j);
	cv::Mat findEssential(int left, int right);
	cv::Mat findEssential(cv::Mat& K_left, cv::Mat& K_right, vector<cv::Point2f>& points_left, vector<cv::Point2f>& points_right);
	void recoverPose(cv::Mat& K_left, cv::Mat& K_right, vector<cv::Point2f>& points_left, vector<cv::Point2f>& points_right, cv::Mat& R, cv::Mat& T);
	int EstimatePose5Point(int left, int right, double* R, double* T);
	/*void findBaselineTriangulation();*/
	void addMoreViewsToReconstruction();
	void BundleAdjustment();
};

#endif // !SFM_H

