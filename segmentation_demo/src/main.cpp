#include "timer.h"
#include "images.h"
#include "mean_shift.h"

#include <iostream>

using namespace images;

bool endsWith(const std::string &string, const std::string &ending)
{
    if (ending.size() > string.size()) {
        return false;
    }
    return std::equal(ending.rbegin(), ending.rend(), string.rbegin(),
                      [](const char a, const char b) {
                          return tolower(a) == tolower(b);
                      }
    );
}

int main(int argc, char **argv)
{
    int sigmaS = 8;
    int sigmaR = 5;
    int minArea = 200;
    SpeedUpLevel speedupLevel = NO_SPEEDUP;

    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <inputImageFilename> <outputImageFilename>" << std::endl;
        return 1;
    }

    std::string inputFilename(argv[1]);
    std::string outputFilename(argv[2]);

    std::cout << "Loading image " << inputFilename << "..." << std::endl;
    Image<unsigned char> image(inputFilename);
    if (image.isNull()) {
        std::cerr << std::endl << "Failed to load image: " << inputFilename << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << image.width << "x" << image.height << "x" << image.cn << std::endl;

    if (image.cn == 4) {
        std::cout << "Image has 4 channels. Alpha channel will be dropped, i.e. segmentation will take into account only RGB part of image (only RGB and grayscale images are supported)." << std::endl;
        image = image.removeAlphaChannel();
    }

    performance_timer timer;
    SegmentedRegions regions = meanShiftSegmentation(image.ptr(), image.width, image.height, image.cn,
                                                     sigmaS, sigmaR, minArea, speedupLevel, true);
    std::cout << "Total segmentation time:\t" << timer.elapsed() << " s" << std::endl;

    Image<unsigned char> imageWithBorders = image.copy();
    for (size_t i = 0; i < regions.getNumRegions(); ++i) {
        std::vector<PixelPosition> border = regions.getRegionBorder(i);
        for (size_t j = 0; j < border.size(); ++j) {
            for (size_t c = 0; c < imageWithBorders.cn; ++c) {
                imageWithBorders(border[j].row, border[j].column, c) = 255;
            }
        }
    }

    if (endsWith(outputFilename, ".jpg") || endsWith(outputFilename, ".jpeg")) {
        imageWithBorders.saveJPEG(outputFilename);
    } else if (endsWith(outputFilename, ".png")) {
        imageWithBorders.savePNG(outputFilename);
    } else {
        std::cerr << "Extension of output image file is not recognized - saving as PNG." << std::endl;
        imageWithBorders.savePNG(outputFilename);
    }

//	ImageWindow window = imageWithBorders.show("Segmented image");
//	while (!window.isClosed()) {
//		window.wait(30);
//	}

    std::cout << "Segmented image saved to " << outputFilename << std::endl;
    return 0;
}
