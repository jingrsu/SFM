#ifndef SFM_H
#define SFM_H
#include <opencv2/core/core.hpp>
#include "imageset.h"
#include "keypoints.h"
#include "matches.h"
#include "tracklist.h"

using namespace std;

class SFM
{
public:
	SFM();

private:

	ImageSet & imageset;
	KeyPoints& keypoints;
	Matches& matches;
	TrackList& tracklist;

	void findBaselineTriangulation();
	void addMoreViewsToReconstruction();
	void BundleAdjustment();
};

#endif // !SFM_H

