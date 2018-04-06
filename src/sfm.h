#ifndef SFM_H
#define SFM_H
#include <opencv2/core/core.hpp>
#include "imageset.h"
#include "keypoints.h"
#include "matches.h"
#include "tracklist.h"
#include <ceres\ceres.h>
#include <ceres\rotation.h>

using namespace std;

struct ReprojectCost
{
	cv::Point2d observation;

	ReprojectCost(cv::Point2d& observation)
		: observation(observation)
	{
	}

	template <typename T>
	bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
	{
		const T* r = extrinsic;
		const T* t = &extrinsic[3];

		T pos_proj[3];
		ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

		// Apply the camera translation
		pos_proj[0] += t[0];
		pos_proj[1] += t[1];
		pos_proj[2] += t[2];

		const T x = pos_proj[0] / pos_proj[2];
		const T y = pos_proj[1] / pos_proj[2];

		const T fx = intrinsic[0];
		const T fy = intrinsic[1];
		const T cx = intrinsic[2];
		const T cy = intrinsic[3];

		// Apply intrinsic
		const T u = fx * x + cx;
		const T v = fy * y + cy;

		residuals[0] = u - T(observation.x);
		residuals[1] = v - T(observation.y);

		return true;
	}
};

class SFM
{
public:
	SFM(ImageSet& _imageset, KeyPoints& _keypoints, Matches& _matches, TrackList& _tracklist);
	void findBaselineTriangulation();
	void addMoreViewsToReconstruction();
	void toBundle2PMVS();
	void toMyPMVS();
private:

	ImageSet & imageset;
	KeyPoints& keypoints;
	Matches& matches;
	TrackList& tracklist;
	set<int> DoneViews;

	void sortPairsForBaseline(vector<pair<float, pair<int, int>>>& pairs);
	void getMatchedPoints(int i, int j, vector<cv::Point2f>&points_i, vector<cv::Point2f>&points_j);
	cv::Mat findEssential(int left, int right);
	cv::Mat findEssential(cv::Mat& K_left, cv::Mat& K_right, vector<cv::Point2f>& points_left, vector<cv::Point2f>& points_right);
	void recoverPose(cv::Mat& K_left, cv::Mat& K_right, vector<cv::Point2f>& points_left, vector<cv::Point2f>& points_right, cv::Mat& R, cv::Mat& T);
	int EstimatePose5Point(int left, int right, double* R, double* T);
	/*void findBaselineTriangulation();*/
	void BundleAdjustment();
};

#endif // !SFM_H

