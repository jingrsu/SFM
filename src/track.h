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
	//������ͶӰ���
	double reprojectionError(const vector<cv::Mat>&pmats, const cv::Mat&points4D);
public:
	//track��Ӧ��״̬
	int status;
	//track��Ӧ��3ά����
	double x;
	double y;
	double z;
	//track��Ӧ����ͶӰ���
	double error;
	//track������������
	vector<Feature> features;
	//rgb��ɫֵ
	int r;
	int g;
	int b;
	Track();
	//���������
	void addFeature(const Feature&f);
	//track���Ƿ��г�ͻ��������
	bool hasConfilct();
	//���ǻ�track
	void triangulate(const vector<cv::Mat>&pmats);
	friend ostream&operator<<(ostream&os, const Track&track);
};
#endif