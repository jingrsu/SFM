#include "track.h"
Track::Track()
{
	this->x = -1;
	this->y = -1;
	this->z = -1;
	this->r = 255;
	this->g = 255;
	this->b = 255;
	this->status = -1;
}
void Track::addFeature(const Feature &f)
{
	features.push_back(f);
}
bool Track::hasConfilct()
{
	set<int> frameIds;

	for (int i = 0; i<features.size(); i++)
	{
		Feature&f1 = features[i];
		if (frameIds.insert(f1.frameIdx).second == false)
			return true;
	}
	return false;
}
double Track::reprojectionError(const vector<cv::Mat>&pmats, const cv::Mat&points4D)
{
	double e = 0;
	for (int i = 0; i<features.size(); i++)
	{
		Feature&f = features[i];
		cv::Mat proPoints = pmats[f.frameIdx] * points4D;
		proPoints = proPoints / proPoints.at<double>(2, 0);
		double ex = f.x - proPoints.at<double>(0, 0);
		double ey = f.y - proPoints.at<double>(1, 0);
		e = e + sqrt(ex*ex + ey * ey);
	}
	return e / features.size();
}
void Track::triangulate(const vector<cv::Mat>&pmats)
{
	double minE = 1e+10;
	double bestX = 0;
	double bestY = 0;
	double bestZ = 0;
	for (int iter = 0; iter<features.size() * 2; iter++) {

		int i = abs(rand()) % features.size();
		int j = abs(rand()) % features.size();
		while (i == j)
		{
			j = abs(rand()) % features.size();
		}
		Feature&f1 = features[i];
		Feature&f2 = features[j];
		cv::Mat p1 = pmats[f1.frameIdx];
		cv::Mat p2 = pmats[f2.frameIdx];
		vector<cv::Point2d> points1, points2;
		points1.push_back(cv::Point2d(f1.x, f1.y));
		points2.push_back(cv::Point2d(f2.x, f2.y));
		cv::Mat points4D;
		triangulatePoints(p1, p2, points1, points2, points4D);
		points4D = points4D / points4D.at<double>(3, 0);
		double e = reprojectionError(pmats, points4D);
		if (e<minE)
		{
			minE = e;
			bestX = points4D.at<double>(0, 0);
			bestY = points4D.at<double>(1, 0);
			bestZ = points4D.at<double>(2, 0);
		}
	}
	//cout<<"×îÐ¡Îó²î:"<<minE<<endl;
	this->x = bestX;
	this->y = bestY;
	this->z = bestZ;
	this->error = minE;
}
ostream&operator<<(ostream&os, const Track&track)
{
	for (vector<Feature>::const_iterator it = track.features.begin(); it != track.features.end(); it++)
	{

		os << it->frameIdx << " " << it->x << " " << it->y << ";";
	}
	return os;
}