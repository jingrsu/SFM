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
	//track�е�һ�����RT����������������
	pair<int, int> frameIdx_and_idx;
	//track��Ӧ��3ά����
	cv::Point3d position;
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
	void triangulate(cv::Mat Pmat1, cv::Mat Pmat2, cv::Point2f point1, cv::Point2f point2);
	friend ostream&operator<<(ostream&os, const Track&track);
};
#endif