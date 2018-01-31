#ifndef tracklist_hpp
#define tracklist_hpp
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <vector>
#include <string>
#include <fstream>
#include "feature.h"
#include "track.h"
#include "keypoints.h"
#include "matches.h"

using namespace std;

class TrackList {
private:
	vector<Track> tracks;
	vector<vector<int> > trackIds;
	void removeTracks(const vector<bool>&invalidTracks, vector<Track>&tracks);
public:
	TrackList(const KeyPoints&keypoints, const Matches&matches);
	TrackList(const string&fileName);
	//���ǻ�
	void triangulate(const vector<cv::Mat>&pmats);
	void getColor(const vector<cv::Mat>&images);
	//���浽ply�ļ�
	void save2ply(const string&fileName);
	void saveTo(const string&fileName);
	void  printf();
	friend ostream&operator<<(ostream&os, const TrackList&trackList);
};
#endif
