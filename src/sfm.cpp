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
			if (matchVec.size() < 200)
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
	cout << "count1: " << count1 << " count2: " << count2 << endl;
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
		T = -T;
	}
	else
	{
		R2.copyTo(R);
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
		for (size_t k = 0; k < matchVec.size(); k++)
		{
			int idx1 = trackIdx1[matchVec[k].queryIdx];
			int idx2 = trackIdx2[matchVec[k].trainIdx];
			if (idx1 == -1 || idx2 == -1 || idx1 != idx2)
				continue;

			cv::Mat_<double> col = s.col(k);
			col /= col(3);

			tracklist.tracks[idx1].position.x = col(0);
			tracklist.tracks[idx1].position.y = col(1);
			tracklist.tracks[idx1].position.z = col(2);

			tracklist.tracks[idx1].status = 1;
			tracklist.DoneTracks++;
		}

		for (size_t k = 0; k < trackIdx1.size(); k++)
		{
			int idx = trackIdx1[k];
			if (idx == -1)
				continue;

			if (tracklist.tracks[idx].status == -1)
			{
				tracklist.tracks[idx].status = 0;
				tracklist.tracks[idx].frameIdx_and_idx = pair<int, int>(i, k);
			}
		}

		for (size_t k = 0; k < trackIdx2.size(); k++)
		{
			int idx = trackIdx2[k];
			if (idx == -1)
				continue;

			if (tracklist.tracks[idx].status == -1)
			{
				tracklist.tracks[idx].status = 0;
				tracklist.tracks[idx].frameIdx_and_idx = pair<int, int>(j, k);
			}
		}

		imageset.updateRTmat(i, cv::Mat::eye(3, 3, CV_32FC1), cv::Mat::zeros(3, 1, CV_32FC1));
		imageset.updateRTmat(j, R, T);
		DoneViews.insert(i);
		DoneViews.insert(j);
		break;

		system("pause");
	}

	BundleAdjustment();
}

void SFM::addMoreViewsToReconstruction()
{
	cout << "----------Add More Views----------" << endl;

	while (DoneViews.size() != imageset.images.size())
	{
		size_t bestNumMatches = 0;
		size_t bestView;

		for (size_t viewIdx = 0; viewIdx < imageset.images.size(); viewIdx++)
		{
			if (DoneViews.find(viewIdx) != DoneViews.end())
				continue;

			size_t tmpNumMatches = 0;
			vector<int> trackIdx = tracklist.trackIds[viewIdx];
			for (size_t i = 0; i < trackIdx.size(); i++)
			{
				if (trackIdx[i] == -1)
					continue;

				if (tracklist.tracks[trackIdx[i]].status == 1)
					tmpNumMatches++;
			}

			if (tmpNumMatches > bestNumMatches)
			{
				bestNumMatches = tmpNumMatches;
				bestView = viewIdx;
			}
		}

		if (bestNumMatches < 10)
		{
			cout << "There are too few matching images here" << endl;
			break;
		}

		DoneViews.insert(bestView);

		cout << "Best view " << bestView << " has " << bestNumMatches << " matches" << endl;
		cout << "Trying to recover the new view camera pose" << endl;

		vector<cv::Point3f> object_points;
		vector<cv::Point2f> image_points;
		vector<int> trackIdx = tracklist.trackIds[bestView];
		assert(trackIdx.size() == keypoints.getKeyPointNum(bestView));
		for (size_t i = 0; i < trackIdx.size(); i++)
		{
			if (trackIdx[i] == -1)
				continue;

			Track &track = tracklist.tracks[trackIdx[i]];
			if (track.status != 1)
				continue;

			object_points.push_back(cv::Point3f(track.position));
			image_points.push_back(keypoints.getKeyPoint(bestView, i).pt);
		}

		cv::Mat r, R, T, inliers;;
		cv::solvePnPRansac(object_points, image_points, imageset.Kmats[bestView], cv::noArray(), r, T, false, 100, 10.0, 0.99, inliers);//之后改进，判断内点比率
		cout << "Inliers ratio is " << (float)cv::countNonZero(inliers) / object_points.size() << endl;
		cv::Rodrigues(r, R);
		imageset.updateRTmat(bestView, R, T);

		cout << "New view " << bestView << " pose " << endl << imageset.RTmats[bestView] << endl;

		for (size_t i = 0; i < trackIdx.size(); i++)
		{
			if (trackIdx[i] == -1)
				continue;

			Track &track = tracklist.tracks[trackIdx[i]];
			if (track.status == 0)
			{
				track.triangulate(imageset.Pmats[track.frameIdx_and_idx.first], imageset.Pmats[bestView], keypoints.getKeyPoint(track.frameIdx_and_idx.first, track.frameIdx_and_idx.second).pt, keypoints.getKeyPoint(bestView, i).pt);
				track.status = 1;
				tracklist.DoneTracks++;
			}
		}

		BundleAdjustment();
	}
}

void SFM::BundleAdjustment()
{
	cout << "----------Try to Bundle Adjustment----------" << endl;
	vector<cv::Mat> extrinsics;
	//vector<cv::Mat> intrinsics;
	cv::Mat intrinsic(cv::Matx41d(imageset.Kmats[*DoneViews.begin()].at<double>(0, 0), imageset.Kmats[*DoneViews.begin()].at<double>(1, 1), imageset.Kmats[*DoneViews.begin()].at<double>(0, 2), imageset.Kmats[*DoneViews.begin()].at<double>(1, 2)));

	set<int>::iterator it = DoneViews.begin();
	while (it != DoneViews.end())
	{
		cv::Mat extrinsic(6, 1, CV_64FC1);
		cv::Mat r;
		Rodrigues(imageset.RTmats[*it](cv::Range(0, 3), cv::Range(0, 3)), r);
		r.copyTo(extrinsic.rowRange(0, 3));
		imageset.RTmats[*it].col(3).copyTo(extrinsic.rowRange(3, 6));
		extrinsics.push_back(extrinsic);

		//cv::Mat intrinsic(cv::Matx41d(imageset.Kmats[*it].at<double>(0, 0), imageset.Kmats[*it].at<double>(1, 1), imageset.Kmats[*it].at<double>(0, 2), imageset.Kmats[*it].at<double>(1, 2)));
		//intrinsics.push_back(intrinsic);

		it++;
	}

	ceres::Problem problem;

	for (size_t i = 0; i < extrinsics.size(); ++i)
	{
		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
	}
	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());
	//for (size_t i = 0; i < intrinsics.size(); ++i)
	//{
	//	problem.AddParameterBlock(intrinsics[i].ptr<double>(), 4);
	//}
	problem.AddParameterBlock(intrinsic.ptr<double>(), 4);
	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);

	it = DoneViews.begin();
	size_t count = 0;
	while (it != DoneViews.end())
	{
		for (size_t i = 0; i < tracklist.trackIds[*it].size(); i++)
		{
			int idx = tracklist.trackIds[*it][i];
			if (idx < 0)
				continue;
			if (tracklist.tracks[idx].status != 1)
				continue;

			cv::Point2d observed = keypoints.getKeyPoint(*it, i).pt;
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));
			problem.AddResidualBlock(
				cost_function,
				loss_function,
				intrinsic.ptr<double>(),            // Intrinsic
				extrinsics[count].ptr<double>(),  // View Rotation and Translation
				&(tracklist.tracks[idx].position.x)          // Point in 3D space
			);
		}

		count++;
		it++;
	}

	// Solve BA
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.minimizer_progress_to_stdout = false;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = 1;
	ceres_config_options.preconditioner_type = ceres::JACOBI;
	ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
	ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);
	if (!summary.IsSolutionUsable())
	{
		std::cout << "Bundle Adjustment failed." << std::endl;
	}
	else
	{
		// Display statistics about the minimization
		std::cout << std::endl
			<< "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << extrinsics.size() << "\n"
			<< " #residuals: " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " Time (s): " << summary.total_time_in_seconds << "\n"
			<< std::endl;
	}

	it = DoneViews.begin();
	count = 0;
	while (it != DoneViews.end())
	{
		//cv::Mat intrinsic = intrinsics[count];

		imageset.Kmats[*it].at<double>(0, 0) = intrinsic.at<double>(0);
		imageset.Kmats[*it].at<double>(1, 1) = intrinsic.at<double>(1);
		imageset.Kmats[*it].at<double>(0, 2) = intrinsic.at<double>(2);
		imageset.Kmats[*it].at<double>(1, 2) = intrinsic.at<double>(3);

		cv::Mat extrinsic = extrinsics[count];
		cv::Mat r, R;

		extrinsic.rowRange(0, 3).copyTo(r);
		Rodrigues(r, R);
		imageset.updateRTmat(*it, R, extrinsic.rowRange(3, 6));

		count++;
		it++;
	}
}

void SFM::toBundle2PMVS()
{
	ofstream fout("bundle.out");

	fout << "# Bundle file v0.3\n";
	fout << imageset.images.size() << ' ' << tracklist.DoneTracks << '\n';
	for (size_t i = 0; i < imageset.images.size(); i++)
	{
		fout << imageset.Kmats[i].at<double>(0, 0) << ' ' << (double)0.0 << ' ' << (double)0.0 << '\n';
		fout << imageset.RTmats[i].at<double>(0, 0) << ' ' << imageset.RTmats[i].at<double>(0, 1) << ' ' << imageset.RTmats[i].at<double>(0, 2) << '\n';
		fout << imageset.RTmats[i].at<double>(1, 0) << ' ' << imageset.RTmats[i].at<double>(1, 1) << ' ' << imageset.RTmats[i].at<double>(1, 2) << '\n';
		fout << imageset.RTmats[i].at<double>(2, 0) << ' ' << imageset.RTmats[i].at<double>(2, 1) << ' ' << imageset.RTmats[i].at<double>(2, 2) << '\n';
		fout << imageset.RTmats[i].at<double>(0, 3) << ' ' << imageset.RTmats[i].at<double>(1, 3) << ' ' << imageset.RTmats[i].at<double>(2, 3) << '\n';
	}

	for (size_t i = 0; i < tracklist.tracks.size(); i++)
	{
		Track &track = tracklist.tracks[i];
		if (track.status != 1)
			continue;

		fout << track.position.x << ' ' << track.position.y << ' ' << track.position.z << '\n';
		fout << track.r << ' ' << track.g << ' ' << track.b << '\n';
		fout << track.features.size() << ' ';
		for (size_t j = 0; j < track.features.size(); j++)
		{
			Feature &feature = track.features[j];
			fout << feature.frameIdx << ' ' << feature.idx << ' ' << feature.x - imageset.images[feature.frameIdx].cols/2 << ' ' << imageset.images[feature.frameIdx].rows / 2 - feature.y;
			if (j == track.features.size() - 1)
				fout << '\n';
			else
				fout << ' ';
		}
	}

	fout.close();

	fout.open("list.txt");

	for (size_t i = 0; i < imageset.imageNames.size(); i++)
	{
		fout << imageset.imageNames[i] << '\n';
	}

	fout.close();
}

void SFM::toMyPMVS()
{
	ofstream fout("imageParameter.txt");

	fout << imageset.images.size() << '\n';
	for (size_t i = 0; i < imageset.images.size(); i++)
	{
		fout << imageset.imageNames[i] << ' '
			<< imageset.Kmats[i].at<double>(0, 0) << ' ' << imageset.Kmats[i].at<double>(0, 1) << ' ' << imageset.Kmats[i].at<double>(0, 2) << ' '
			<< imageset.Kmats[i].at<double>(1, 0) << ' ' << imageset.Kmats[i].at<double>(1, 1) << ' ' << imageset.Kmats[i].at<double>(1, 2) << ' '
			<< imageset.Kmats[i].at<double>(2, 0) << ' ' << imageset.Kmats[i].at<double>(2, 1) << ' ' << imageset.Kmats[i].at<double>(2, 2) << ' '
			<< imageset.RTmats[i].at<double>(0, 0) << ' ' << imageset.RTmats[i].at<double>(0, 1) << ' ' << imageset.RTmats[i].at<double>(0, 2) << ' '
			<< imageset.RTmats[i].at<double>(1, 0) << ' ' << imageset.RTmats[i].at<double>(1, 1) << ' ' << imageset.RTmats[i].at<double>(1, 2) << ' '
			<< imageset.RTmats[i].at<double>(2, 0) << ' ' << imageset.RTmats[i].at<double>(2, 1) << ' ' << imageset.RTmats[i].at<double>(2, 2) << ' '
			<< imageset.RTmats[i].at<double>(0, 3) << ' ' << imageset.RTmats[i].at<double>(1, 3) << ' ' << imageset.RTmats[i].at<double>(2, 3) << ' '
			<< '\n';
	}
	fout.close();
}