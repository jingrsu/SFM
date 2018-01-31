#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "imageset.h"
#include "keypoints.h"
#include "matches.h"
#include "tracklist.h"

int main()
{
	//string imageDir = "..\\images";
	//ImageSet imageset(imageDir);
	//KeyPoints keypoints(imageset.images);
	//keypoints.saveTo("..\\keypoints.txt");
	//Matches matches(keypoints);
	//matches.saveTo("..\\matches.txt");
	KeyPoints keypoints("..\\keypoints.txt");
	Matches matches("..\\matches.txt");
	TrackList tracklist(keypoints, matches);
	//tracklist.printf();
	system("pause");

	return 0;
}