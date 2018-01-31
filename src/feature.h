#ifndef Feature_hpp
#define Feature_hpp
class Feature {
public:
	int frameIdx; //视图编号
	int idx; //视图内的序号
	double x; //x坐标
	double y; //y坐标

	Feature(int frameIdx, int idx, double x, double y);
	Feature();

};
#endif