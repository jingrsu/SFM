#ifndef Feature_hpp
#define Feature_hpp
class Feature {
public:
	int frameIdx; //��ͼ���
	int idx; //��ͼ�ڵ����
	double x; //x����
	double y; //y����

	Feature(int frameIdx, int idx, double x, double y);
	Feature();

};
#endif