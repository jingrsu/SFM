#include "matches.h"

//用基础矩阵计算匹配
void Matches::pairwiseMatch(const vector<cv::KeyPoint>& keypoints1, const vector<cv::KeyPoint>& keypoints2, const cv::Mat& descriptors1, const cv::Mat& descriptors2, vector<cv::DMatch>& matches)
{
	vector<cv::DMatch> initalMatches;
	cv::BFMatcher bf(cv::NORM_L2, true);
	bf.match(descriptors1, descriptors2, initalMatches);
	//计算基础矩阵
	vector<cv::Point2d> points1;
	vector<cv::Point2d> points2;
	for (vector<cv::DMatch>::iterator it = initalMatches.begin(); it != initalMatches.end(); it++)
	{
		points1.push_back(keypoints1[it->queryIdx].pt);
		points2.push_back(keypoints2[it->trainIdx].pt);
	}
	vector<uchar> mask;
	cv::Mat F = findFundamentalMat(points1, points2, cv::FM_RANSAC, 1, 0.99, mask);
	//用ransac算法产生的mask获取更好的特征匹配
	for (vector<cv::DMatch>::size_type i = 0; i<initalMatches.size(); i++)
	{
		if (mask[i])
			matches.push_back(initalMatches[i]);
	}
}

Matches::Matches(const KeyPoints& keyPoints)
{
	int nFrames = keyPoints.getFrameNum();
	matchesTable.resize(nFrames, vector<vector<cv::DMatch> >(nFrames, vector<cv::DMatch>()));
	for (int i = 0; i<nFrames; i++)
		for (int j = i + 1; j<nFrames; j++)
		{
			cout << "计算视图" << i << "与视图" << j << "之间特征点的匹配,";
			pairwiseMatch(keyPoints.getKeyPoints(i), keyPoints.getKeyPoints(j), keyPoints.getDescriptors(i), keyPoints.getDescriptors(j), matchesTable[i][j]);
			cout << "匹配点数为" << matchesTable[i][j].size() << endl;
		}
}

Matches::Matches(const string& fileName)
{
	ifstream ifs(fileName.c_str());
	int nFrames;
	ifs >> nFrames;
	matchesTable.resize(nFrames, vector<vector<cv::DMatch> >(nFrames, vector<cv::DMatch>()));
	for (int i = 0; i<nFrames; i++)
		for (int j = i + 1; j<nFrames; j++)
		{
			int size;
			ifs >> size;
			for (int k = 0; k<size; k++) {
				cv::DMatch match;
				ifs >> match.queryIdx >> match.trainIdx;
				matchesTable[i][j].push_back(match);
			}
		}
	ifs.close();
}

void Matches::saveTo(const string & fileName)
{
	ofstream ofs(fileName.c_str());
	int nFrames = (int)matchesTable.size();
	ofs << nFrames << endl;
	for (int i = 0; i<nFrames; i++)
		for (int j = i + 1; j<nFrames; j++)
		{
			ofs << matchesTable[i][j].size() << " ";
			vector<cv::DMatch>& matches = matchesTable[i][j];
			for (int k = 0; k < matchesTable[i][j].size(); k++)
			{
				if (k == matchesTable[i][j].size() - 1)
				{
					ofs << matches[k].queryIdx << " " << matches[k].trainIdx << endl;
				}
				else
				{
					ofs << matches[k].queryIdx << " " << matches[k].trainIdx << " ";
				}


			}
		}
	ofs.close();
}

const vector<cv::DMatch>& Matches::getMatches(int i, int j)const
{
	return matchesTable[i][j];
}