#include "sfm.h"

SFM::SFM(ImageSet& _imageset, KeyPoints& _keypoints, Matches& _matches, TrackList& _tracklist)
	:imageset(_imageset), keypoints(_keypoints), matches(_matches), tracklist(_tracklist)
{
	;
}

void SFM::getMatchedPoints(int i, int j, vector<cv::Point2f>&points_i, vector<cv::Point2f>&points_j)
{
	points_i.clear();
	points_j.clear();
	const vector<cv::KeyPoint>& keypoint_i = keypoints.getKeyPoints(i);
	const vector<cv::KeyPoint>& keypoint_j = keypoints.getKeyPoints(j);
	const vector<cv::DMatch>& matchVec = matches.getMatches(i, j);
	for (size_t i = 0; i < matchVec.size(); i++)
	{
		points_i.push_back(keypoint_i[matchVec[i].queryIdx].pt);
		points_j.push_back(keypoint_j[matchVec[i].trainIdx].pt);
	}
}

void SFM::sortPairsForBaseline(vector<pair<float, pair<int, int>>>& pairs)
{
	const int nFrames = keypoints.getFrameNum();
	for (size_t i = 0; i < nFrames - 1; i++)
	{
		for (size_t j = i + 1; j < nFrames; j++)
		{
			const vector<cv::DMatch>& matchVec = matches.getMatches(i, j);
			if (matchVec.size() < 100)
				continue;
			vector<cv::Point2f> points_i, points_j;
			cv::Mat mask;
			getMatchedPoints(i, j, points_i, points_j);
			cv::findHomography(points_i, points_j, mask);
			float inliersRatio = float(cv::countNonZero(mask)) / points_i.size();
			pairs.push_back(make_pair(inliersRatio, make_pair(i, j)));

			cout << "Homography inliers ratio: " << i << ", " << j << " " << endl;
		}
	}
	if (pairs.empty())
	{
		cout << "----------Can not find a good Baseline----------" << endl
			<< "Fuck you man!!! Your image quality is very low" << endl;
	}
	sort(pairs.begin(), pairs.end());
}

cv::Mat SFM::findEssential(int left, int right)
{
	vector<cv::Point2f> points_left, points_right;
	cv::Mat mask;
	getMatchedPoints(left, right, points_left, points_right);
	cv::Mat F = cv::findFundamentalMat(points_left, points_right, cv::FM_RANSAC, 1, 0.99, mask);
	cv::Mat E = imageset.Kmats[left].t()*F*imageset.Kmats[right];

	return	E;
}

cv::Mat SFM::findEssential(cv::Mat& K_left, cv::Mat& K_right, vector<cv::Point2f>& points_left, vector<cv::Point2f>& points_right)
{
	cv::Mat mask;
	cv::Mat F = cv::findFundamentalMat(points_left, points_right, cv::FM_RANSAC, 1, 0.99, mask);
	cv::Mat E = K_left.t() * F * K_right;

	return	E;
}

void SFM::recoverPose(cv::Mat& K_left, cv::Mat& K_right, vector<cv::Point2f>& points_left, vector<cv::Point2f>& points_right, cv::Mat& R, cv::Mat& T)
{
	cv::Mat E = findEssential(K_left, K_right, points_left, points_right);
	cv::Mat U, S, Vt, R1, R2;
	cv::SVD::compute(E, S, U, Vt);

	U.col(2).copyTo(T);
	T /= cv::norm(T);

	cv::Mat W(3, 3, CV_64F, cv::Scalar(0));
	W.at<double>(0, 1) = -1;
	W.at<double>(1, 0) = 1;
	W.at<double>(2, 2) = 1;

	R1 = U * W * Vt;
	if (cv::determinant(R1) < 0)
		R1 = -R1;

	R2 = U * W.t() * Vt;
	if (cv::determinant(R2) < 0)
		R2 = -R2;

	cout << "----------Try to find the correct [R|T]----------" << endl;

	cv::Mat proj1(3, 4, CV_64FC1,cv::Scalar(0));
	cv::Mat proj2(3, 4, CV_64FC1);
	cv::Mat I = cv::Mat::eye(3, 3, CV_64FC1);

	I.convertTo(proj1(cv::Range(0, 3), cv::Range(0, 3)), CV_64FC1);
	R1.convertTo(proj2(cv::Range(0, 3), cv::Range(0, 3)), CV_64FC1);
	T.convertTo(proj2.col(3), CV_64FC1);

	proj1 = K_left * proj1;
	proj2 = K_right * proj2;

	cv::Mat s;
	cv::triangulatePoints(proj1, proj2, points_left, points_right, s);

	int count1 = 0, count2 = 0;
	for (size_t i = 0; i < s.cols; i++)
	{
		cv::Mat_<double> col = s.col(i);
		col /= col(3);
		cv::Mat_<double> col2 = R1 * col.rowRange(0, 3) + T;

		if (col(2) > 0)
			count1++;
		else
			count1--;
		
		if (col2(2) > 0)
			count2++;
		else
			count2--;
	}

	if (count1 > 0 && count2 > 0)
	{
		R1.copyTo(R);
	}
	else if (count1 < 0 && count2 < 0)
	{
		R1.copyTo(R);
		T = -T;
	}
	else if (count1 < 0 && count2 > 0)
	{
		R2.copyTo(R);
	}
	else
	{
		R2.copyTo(R);
		T = -T;
	}
}

//int SFM::EstimatePose5Point(int left, int right, double* R, double* T)
//{
//	const vector<cv::DMatch>& matchVec = matches.getMatches(left, right);
//	int nFrames = matchVec.size();
//
//	v2_t* k1_pts = new v2_t[nFrames];
//	v2_t* k2_pts = new v2_t[nFrames];
//	const vector<cv::KeyPoint>& points1 = keypoints.getKeyPoints(left);
//	const vector<cv::KeyPoint>& points2 = keypoints.getKeyPoints(right);
//
//	for (size_t i = 0; i < nFrames; i++)
//	{
//		cv::Point2f pt1 = points1[matchVec[i].queryIdx].pt;
//		cv::Point2f pt2 = points2[matchVec[i].trainIdx].pt;
//
//		k1_pts[i] = { {pt1.x,pt1.y} };
//		k2_pts[i] = { {pt2.x,pt2.y} };
//	}
//
//	double K1[9], K2[9];
//	K1[0] = imageset.Kmats[left].at<double>(0, 0); K1[1] = imageset.Kmats[left].at<double>(0, 1); K1[2] = imageset.Kmats[left].at<double>(0, 2);
//	K1[3] = imageset.Kmats[left].at<double>(1, 0); K1[4] = imageset.Kmats[left].at<double>(1, 1); K1[5] = imageset.Kmats[left].at<double>(1, 2);
//	K1[6] = imageset.Kmats[left].at<double>(2, 0); K1[7] = imageset.Kmats[left].at<double>(2, 1); K1[8] = imageset.Kmats[left].at<double>(2, 2);
//
//	K2[0] = imageset.Kmats[right].at<double>(0, 0); K2[1] = imageset.Kmats[right].at<double>(0, 1); K2[2] = imageset.Kmats[right].at<double>(0, 2);
//	K2[3] = imageset.Kmats[right].at<double>(1, 0); K2[4] = imageset.Kmats[right].at<double>(1, 1); K2[5] = imageset.Kmats[right].at<double>(1, 2);
//	K2[6] = imageset.Kmats[right].at<double>(2, 0); K2[7] = imageset.Kmats[right].at<double>(2, 1); K2[8] = imageset.Kmats[right].at<double>(2, 2);
//
//	int num_inliers = compute_pose_ransac(nFrames, k1_pts, k2_pts,
//		K1, K2, 2.0, 1024, R, T);
//
//	return num_inliers;
//}

void SFM::findBaselineTriangulation()
{
	cout << "----------Find Baseline Triangulation----------" << endl;

	int num_inliers = 0;
	cv::Mat R, T;
	vector<pair<float, pair<int, int>>> pairsHomographyInliers;
	sortPairsForBaseline(pairsHomographyInliers);
	for (size_t k = 0; k < pairsHomographyInliers.size(); k++)
	{
		int i = pairsHomographyInliers[k].second.first;
		int j = pairsHomographyInliers[k].second.second;

		vector<cv::Point2f> points_left, points_right;
		getMatchedPoints(i, j, points_left, points_right);

		cout << "Try to transform " << i << ", " << j << " ..." << endl;
		recoverPose(imageset.Kmats[i], imageset.Kmats[j], points_left, points_right, R, T);

		//cout << "Found " << num_inliers << " inliers, the ratio is " << float(num_inliers) / matches.getMatches(i, j).size();
		cout << "----------Try to triangulate----------" << endl;

		cv::Mat proj1(3, 4, CV_64FC1, cv::Scalar(0));
		cv::Mat proj2(3, 4, CV_64FC1);
		cv::Mat I = cv::Mat::eye(3, 3, CV_64FC1);

		I.convertTo(proj1(cv::Range(0, 3), cv::Range(0, 3)), CV_64FC1);
		R.convertTo(proj2(cv::Range(0, 3), cv::Range(0, 3)), CV_64FC1);
		T.convertTo(proj2.col(3), CV_64FC1);

		proj1 = imageset.Kmats[i] * proj1;
		proj2 = imageset.Kmats[j] * proj2;

		cv::Mat s;
		cv::triangulatePoints(proj1, proj2, points_left, points_right, s);

		const vector<cv::DMatch>& matchVec = matches.getMatches(i, j);
		vector<int> trackIdx1 = tracklist.trackIds[i];
		vector<int> trackIdx2 = tracklist.trackIds[j];
		for (size_t k = 0; k < matchVec.size; k++)
		{
			int idx1 = trackIdx1[matchVec[k].queryIdx];
			int idx2 = trackIdx2[matchVec[k].trainIdx];
			if (idx1 == -1 || idx2 == -1 || idx1 != idx2)
				continue;

			cv::Mat_<double> col = s.col(k);
			col /= col(3);

			tracklist.tracks[idx1].x = col(0);
			tracklist.tracks[idx1].y = col(1);
			tracklist.tracks[idx1].z = col(2);
		}

		system("pause");
	}
}