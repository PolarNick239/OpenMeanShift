#include "timer.h"
#include "mean_shift.h"
#include "msImageProcessor.h"

#include <stdexcept>
#include <iostream>

void SegmentedRegions::init(size_t width, size_t height, const RegionList &regionList, const int* labels)
{
    this->width = width;
    this->height = height;

    borders = std::vector<std::vector<PixelPosition> >(regionList.GetNumRegions());
    for (size_t i = 0; i < borders.size(); ++i) {
        borders[i].resize(regionList.GetRegionCount(i));
        for (size_t j = 0; j < borders[i].size(); ++j) {
            int index = regionList.GetRegionIndeces(i)[j];
            PixelPosition p;
            p.row = index / width;
            p.column = index % width;
            borders[i][j] = p;
        }
    }

    nlabels = 0;
    pixelsLabels.resize(height * width);
    for (ptrdiff_t j = 0; j < height; ++j) {
        for (ptrdiff_t i = 0; i < width; ++i) {
            pixelsLabels[j * width + i] = labels[j * width + i];
            nlabels = std::max(nlabels, labels[j * width + i] + 1);
        }
    }
}

size_t SegmentedRegions::getNumRegions()
{
    return borders.size();
}

std::vector<PixelPosition> SegmentedRegions::getRegionBorder(size_t regionIndex)
{
    return borders[regionIndex];
}

const std::vector<int>& SegmentedRegions::getPixelLabels()
{
    return pixelsLabels;
}

int SegmentedRegions::getPixelLabelsNumber()
{
    return nlabels;
}

SegmentedRegions meanShiftSegmentation(const unsigned char *data, int width, int height, int nChannels,
                                       float sigmaS, float sigmaR, int minRegion, SpeedUpLevel implementation,
                                       bool verbose)
{
    msImageProcessor processor;

    if (nChannels == 3) {
        processor.DefineImage(data, COLOR, height, width);
    } else if (nChannels == 1) {
        processor.DefineImage(data, GRAYSCALE, height, width);
    } else {
        throw std::runtime_error("Only grayscale and 3-channels images are supported!");
    }

    performance_timer timer_filter;
    processor.Filter(sigmaS, sigmaR, implementation);
    if (processor.ErrorStatus) {
        throw std::runtime_error("Filtering failed!");
    }
    if (verbose) {
        std::cout << "Filter completed in\t\t\t" << timer_filter.elapsed() << " s" << std::endl;
    }

    performance_timer fusion_timer;
    processor.FuseRegions(sigmaR, minRegion);
    if (processor.ErrorStatus) {
        throw std::runtime_error("Regions fusion failed!");
    }
    if (verbose) {
        std::cout << "Regions fused in\t\t\t" << fusion_timer.elapsed() << " s" << std::endl;
    }

    SegmentedRegions regions;
    regions.init(width, height, *processor.GetBoundaries(), (const int *) processor.labels);
    return regions;
}
