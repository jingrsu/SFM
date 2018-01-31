#include "tracklist.h"
//数据量大时每调用一次vector的erase都会导致重新分配内存，效率很低
//故此需要减少erase的次数
void TrackList::removeTracks(const vector<bool>&invalidTracks, vector<Track>&tracks)
{
	int j = 0;
	for (int i = 0; i<tracks.size(); i++)
	{
		if (invalidTracks[i])
		{
			vector<Feature>& f = tracks[i].features;
			for (size_t count = 0; count < f.size(); count++)
			{
				trackIds[f[count].frameIdx][f[count].idx] = -1;
			}
		}
		else
		{
			if (i != j)
			{
				tracks[j] = tracks[i];

				vector<Feature>& f = tracks[i].features;
				for (size_t count = 0; count < f.size(); count++)
				{
					trackIds[f[count].frameIdx][f[count].idx] = j;
				}
			}
			j++;
		}
	}
	tracks.erase(tracks.begin() + j, tracks.end());
}
TrackList::TrackList(const KeyPoints&keypoints, const Matches&matches)
{
	int nFrames = keypoints.getFrameNum();
	//用于记录每个视图上每个特征点对应的trackId
	trackIds.resize(nFrames, vector<int>());
	for (int i = 0; i<nFrames; i++)
	{
		for (int j = 0; j<keypoints.getKeyPointNum(i); j++)
		{
			trackIds[i].push_back(-1); //trackIds[i][j]表示第i视图上第j个特征点对应的trackId
		}
	}
	for (int i = 0; i<nFrames; i++) {
		cout << "正在计算视图" << i << "..." << endl;
		for (int j = i + 1; j<nFrames; j++)
		{

			const vector<cv::DMatch>& matchVec = matches.getMatches(i, j);
			if (matchVec.size()>100)
			{
				for (vector<cv::DMatch>::const_iterator it = matchVec.begin(); it != matchVec.end(); it++) {
					int trackId1 = trackIds[i][it->queryIdx];
					int trackId2 = trackIds[j][it->trainIdx];
					cv::Point2d p1 = keypoints.getKeyPoint(i, it->queryIdx).pt;
					cv::Point2d p2 = keypoints.getKeyPoint(j, it->trainIdx).pt;
					if (trackId1 == -1 && trackId2 == -1) {
						//new track，新增一个track，新track的编号为tracks.size()
						trackIds[i][it->queryIdx] = (int)tracks.size();
						trackIds[j][it->trainIdx] = (int)tracks.size();
						Track track;
						track.addFeature(Feature(i, it->queryIdx, p1.x, p1.y));
						track.addFeature(Feature(j, it->trainIdx, p2.x, p2.y));
						tracks.push_back(track);
					}
					else if (trackId1 != -1 && trackId2 == -1)
					{
						trackIds[j][it->trainIdx] = trackId1;
						Track &track = tracks[trackId1];
						track.addFeature(Feature(j, it->trainIdx, p2.x, p2.y));

					}
					else if (trackId1 == -1 && trackId2 != -1)
					{
						trackIds[i][it->queryIdx] = trackId2;
						Track &track = tracks[trackId2];
						track.addFeature(Feature(i, it->queryIdx, p1.x, p1.y));
					}
					else if (trackId1 != trackId2)
					{
						//merge track
						trackIds[j][it->trainIdx] = trackId1;
						Track &track1 = tracks[trackId1];
						Track &track2 = tracks[trackId2];
						for (int k = 0; k<track2.features.size(); k++) {
							Feature&f = track2.features[k];
							track1.addFeature(f);
							trackIds[f.frameIdx][f.idx] = trackId1;
						}
						//不能直接将track2从tracks中移除，如果这样做，trackIds中记录的trackId就会出错
						track2.features.clear();
					}

				}
			}
		}
	}
	//过滤掉冲突的特征点，过滤掉空的track
	vector<bool> invalidTracks(tracks.size(), false);
	for (int i = 0; i<tracks.size(); i++)
	{
		Track&track = tracks[i];
		//删除小于3个视图的，空视图的，有冲突的
		if (track.features.size()<3 || track.features.empty() || track.hasConfilct())
		{
			invalidTracks[i] = true;
			cout << "remove track:" << i << endl;
		}
	}
	removeTracks(invalidTracks, tracks);
}

TrackList::TrackList(const string&fileName)
{
	ifstream ifs(fileName);
	int n;
	ifs >> n;
	tracks.resize(n);
	for (int i = 0; i<n; i++) {
		Track&track = tracks[i];
		ifs >> track.x >> track.y >> track.z;
		int fn;
		ifs >> fn;
		track.features.resize(fn);
		for (int j = 0; j<track.features.size(); j++)
		{
			Feature&f = track.features[j];
			ifs >> f.frameIdx >> f.x >> f.y;
		}
	}
	ifs.close();
}

void TrackList::triangulate(const vector<cv::Mat>&pmats)
{
	for (int i = 0; i<tracks.size(); i++) {
		tracks[i].triangulate(pmats);
	}
	//过滤重投影误差过大的track
	vector<bool> invalidTracks(tracks.size(), false);
	for (int i = 0; i<tracks.size(); i++)
	{
		Track&track = tracks[i];
		if (track.error>2)
		{
			invalidTracks[i] = true;
		}
	}
	removeTracks(invalidTracks, tracks);
}

void TrackList::save2ply(const string&fileName)
{
	ofstream fout(fileName.c_str());

	fout << "ply" << endl;
	fout << "format ascii 1.0" << endl;
	fout << "element vertex " << tracks.size() << endl;
	fout << "property float x" << endl;
	fout << "property float y" << endl;
	fout << "property float z" << endl;
	fout << "property uchar diffuse_red" << endl;
	fout << "property uchar diffuse_green" << endl;
	fout << "property uchar diffuse_blue" << endl;
	fout << "end_header" << endl;
	for (int i = 0; i<tracks.size(); i++) {
		fout << tracks[i].x << " " << tracks[i].y << " " << tracks[i].z << " " << tracks[i].r << " " << tracks[i].g << " " << tracks[i].b << endl;
	}
	fout.close();
}

void TrackList::getColor(const vector<cv::Mat> &images)
{
	for (int i = 0; i<tracks.size(); i++)
	{
		Track &track = tracks[i];
		int r = 0, g = 0, b = 0;
		for (int j = 0; j<track.features.size(); j++)
		{
			Feature &f = track.features[j];
			b += images[f.frameIdx].at<cv::Vec3b>(f.y, f.x)[0];
			g += images[f.frameIdx].at<cv::Vec3b>(f.y, f.x)[1];
			r += images[f.frameIdx].at<cv::Vec3b>(f.y, f.x)[2];
		}
		track.r = r / track.features.size();
		track.g = g / track.features.size();
		track.b = b / track.features.size();
	}
}

void TrackList::saveTo(const string &fileName)
{
	ofstream ofs(fileName);
	ofs << tracks.size() << endl;
	for (int i = 0; i<tracks.size(); i++)
	{
		Track&track = tracks[i];
		ofs << track.x << " " << track.y << " " << track.z << " ";
		ofs << track.features.size() << " ";
		for (int j = 0; j<track.features.size(); j++)
		{
			Feature&f = track.features[j];
			if (j == track.features.size() - 1)
			{
				ofs << f.frameIdx << " " << f.x << " " << f.y << endl;
			}
			else
			{
				ofs << f.frameIdx << " " << f.x << " " << f.y << " ";
			}
		}
	}
	ofs.close();
}

ostream&operator<<(ostream&os, const TrackList&trackList)
{
	for (int i = 0; i<trackList.tracks.size(); i++)
	{
		cout << "(" << trackList.tracks[i].x << "," << trackList.tracks[i].y << "," << trackList.tracks[i].z << ");" << trackList.tracks[i] << endl;

	}
	return os;
}

void TrackList::printf()
{
	for (size_t i = 0; i < tracks.size(); i++)
	{
		cout << "track " << i << ":";
		for (size_t j = 0; j < tracks[i].features.size(); j++)
		{
			cout << tracks[i].features[j].frameIdx << "-" << tracks[i].features[j].idx << "  ";
		}
		cout << endl;
	}

	system("pause");
	for (size_t i = 0; i < trackIds.size(); i++)
	{
		cout << "trackIds " << i << ":";
		for (size_t j = 0; j < trackIds[i].size(); j++)
		{
			cout << trackIds[i][j] << " ";
		}
		cout << endl;
	}
}