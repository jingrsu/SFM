#include "keypoints.h"

KeyPoints::KeyPoints(const vector<cv::Mat>&images) :keypointsVec(images.size(), vector<cv::KeyPoint>()), descriptorsVec(images.size(), cv::Mat())
{
	cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create();
	for (int i = 0; i<images.size(); i++) {
		cout << "计算视图" << i << "的特征点，";

		sift->detectAndCompute(images[i], cv::noArray(), keypointsVec[i], descriptorsVec[i]);
		cout << "特征点共有" << keypointsVec[i].size() << "个" << endl;
	}
}

KeyPoints::KeyPoints(const string&fileName)
{
	ifstream ifs(fileName.c_str());
	int nImages;
	ifs >> nImages;
	keypointsVec.resize(nImages, vector<cv::KeyPoint>());
	descriptorsVec.resize(nImages);
	for (int i = 0; i<nImages; i++)
	{
		int nPoints;
		ifs >> nPoints;
		descriptorsVec[i] = cv::Mat(nPoints, 128, CV_32F);
		for (int j = 0; j<nPoints; j++) {
			cv::KeyPoint point;
			cv::Mat descriptor = descriptorsVec[i].row(j);
			ifs >> point.pt.x >> point.pt.y;
			for (int k = 0; k<descriptor.cols; k++) {
				ifs >> descriptor.at<float>(0, k);
			}
			keypointsVec[i].push_back(point);
		}
	}
	ifs.close();
}

void KeyPoints::saveTo(string fileName)
{
	ofstream ofs(fileName.c_str());
	ofs << keypointsVec.size() << endl;
	for (int i = 0; i<keypointsVec.size(); i++) {
		ofs << keypointsVec[i].size() << endl;
		for (int j = 0; j<keypointsVec[i].size(); j++) {
			ofs << keypointsVec[i][j].pt.x << " " << keypointsVec[i][j].pt.y << " ";
			cv::Mat descriptor = descriptorsVec[i].row(j);
			for (int k = 0; k<descriptor.cols; k++) {
				if (k == descriptor.cols - 1)
					ofs << descriptor.at<float>(0, k) << endl;
				else
					ofs << descriptor.at<float>(0, k) << " ";
			}
		}
	}
	ofs.close();
}
int KeyPoints::getFrameNum()const
{
	return (int)keypointsVec.size();
}
int KeyPoints::getKeyPointNum(int i)const
{
	return (int)keypointsVec[i].size();
}
cv::KeyPoint KeyPoints::getKeyPoint(int i, int j) const
{
	return keypointsVec[i][j];
}
const vector<cv::KeyPoint>& KeyPoints::getKeyPoints(int i) const
{
	return keypointsVec[i];
}
const cv::Mat& KeyPoints::getDescriptors(int i) const
{
	return descriptorsVec[i];
}