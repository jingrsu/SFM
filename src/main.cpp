#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "imageset.h"
#include "keypoints.h"
#include "matches.h"
#include "tracklist.h"
#include "sfm.h"

int main()
{
	string imageDir = "..\\images";
	ImageSet imageset(imageDir);
	KeyPoints keypoints(imageset.images);
	keypoints.saveTo("..\\keypoints.txt");
	//KeyPoints keypoints("..\\keypoints.txt");
	Matches matches(keypoints);
	matches.saveTo("..\\matches.txt");
	//Matches matches("..\\matches.txt");
	TrackList tracklist(keypoints, matches);
	//tracklist.printf();
	SFM sfm(imageset, keypoints, matches, tracklist);
	sfm.findBaselineTriangulation();
	sfm.addMoreViewsToReconstruction();
	tracklist.getColor(imageset.images);
	tracklist.save2yml("structure.yml",imageset);
	tracklist.save2ply("structure.ply");

	//cout << imageset.RTmats[1] << endl << imageset.RTmats[2] << endl;

	system("pause");

	return 0;
}