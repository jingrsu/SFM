#ifndef Track_hpp
#define Track_hpp
#include<opencv2/opencv.hpp>
#include "feature.h"
//#include <vector>
//#include <unordered_map>
#include <iostream>
//#include <set>

using namespace std;

class Track {
private:
	//计算重投影误差
	double reprojectionError(const vector<cv::Mat>&pmats, const cv::Mat&points4D);
public:
	//track对应的状态
	int status;
	//track对应的3维坐标
	double x;
	double y;
	double z;
	//track对应的重投影误差
	double error;
	//track包含的特征点
	vector<Feature> features;
	//rgb颜色值
	int r;
	int g;
	int b;
	Track();
	//添加特征点
	void addFeature(const Feature&f);
	//track中是否有冲突的特征点
	bool hasConfilct();
	//三角化track
	void triangulate(const vector<cv::Mat>&pmats);
	friend ostream&operator<<(ostream&os, const Track&track);
};
#endif