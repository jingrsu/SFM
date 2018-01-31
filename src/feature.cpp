#include "feature.h"

Feature::Feature(int frameIdx, int idx, double x, double y)
{
	this->frameIdx = frameIdx;
	this->idx = idx;
	this->x = x;
	this->y = y;
}
Feature::Feature()
{
	this->frameIdx = -1;
	this->idx = -1;
	this->x = -1;
	this->y = -1;
}