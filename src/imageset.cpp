#include "imageset.h"
#include <tinydir.h>

void get_file_names(string dir_name, vector<string> & names)
{
	names.clear();
	tinydir_dir dir;
	tinydir_open(&dir, dir_name.c_str());

	while (dir.has_next)
	{
		tinydir_file file;
		tinydir_readfile(&dir, &file);
		if (!file.is_dir)
		{
			names.push_back(file.path);
		}
		tinydir_next(&dir);
	}
	tinydir_close(&dir);
}

ImageSet::ImageSet(const string& imageDir, const string& parFileName)
{
	ifstream fin((imageDir + parFileName).c_str());
	int n;
	fin >> n;
	for (int i = 0; i<n; i++)
	{
		string name;
		cv::Mat_<double> k(3, 3);
		cv::Mat_<double> rt(3, 4);
		fin >> name >> k(0, 0) >> k(0, 1) >> k(0, 2) >> k(1, 0) >> k(1, 1) >> k(1, 2) >> k(2, 0) >> k(2, 1) >> k(2, 2) >> 
		rt(0, 0) >> rt(0, 1) >> rt(0, 2) >> rt(1, 0) >> rt(1, 1) >> rt(1, 2) >> rt(2, 0) >> rt(2, 1) >> rt(2, 2) >> rt(0, 3) >> rt(1, 3) >> rt(2, 3);
		imageNames.push_back(name);
		Kmats.push_back(k);
		RTmats.push_back(rt);
		Pmats.push_back(k*rt);
		images.push_back(cv::imread(imageDir + name));
	}
	fin.close();
}

ImageSet::ImageSet(const string& imageDir)
{
	vector<string> names;
	get_file_names(imageDir, names);
	cv::Mat K(cv::Matx33d(
		2759.48, 0, 912.0,
		0, 2764.16, 600.0,
		0, 0, 1));
	cv::Mat M(3, 4, CV_64FC1, cv::Scalar(0));
	for (size_t i = 0; i < names.size(); i++)
	{
		images.push_back(cv::imread(names[i]));
		imageNames.push_back(names[i]);
		Kmats.push_back(K);
		RTmats.push_back(M);
		Pmats.push_back(M);
	}
}

void ImageSet::updateKmat(int i, cv::Mat K)
{
	Kmats[i] = K;

	updatePmat(i);
}

void ImageSet::updateRTmat(int i, cv::Mat R, cv::Mat T)
{
	cv::Mat RT(3, 4, CV_64FC1);
	R.convertTo(RT(cv::Range(0, 3), cv::Range(0, 3)), CV_64FC1);
	T.convertTo(RT.col(3), CV_64FC1);
	RTmats[i] = RT;

	updatePmat(i);
}

void ImageSet::updateRTmat(int i, cv::Mat RT)
{
	RTmats[i] = RT;

	updatePmat(i);
}

void ImageSet::updatePmat(int i)
{
	cv::Mat P = Kmats[i] * RTmats[i];
	Pmats[i] = P;

}