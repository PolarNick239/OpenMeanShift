#pragma once

#include "../segm/tdef.h"

#include <vector>
#include <cstddef>

struct PixelPosition {
    int row;
    int column;
};

class RegionList;

class SegmentedRegions {
public:
    SegmentedRegions() : width(0), height(0) {}

    void init(size_t width, size_t height, const RegionList &regionList);

    size_t getNumRegions();

    std::vector<PixelPosition> getRegionBorder(size_t regionIndex);

private:
    std::vector<std::vector<PixelPosition> > borders;

    size_t height;
    size_t width;
};

SegmentedRegions meanShiftSegmentation(const unsigned char *data, int width, int height, int nChannels,
                                       float sigmaS, float sigmaR, int minRegion,
                                       SpeedUpLevel implementation = HIGH_SPEEDUP,
                                       bool verbose = false
);
