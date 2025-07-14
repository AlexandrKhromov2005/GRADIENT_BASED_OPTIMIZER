#include "../include/embending_and_extracting.h"

void embend(const std::string& imagePath, const std::string& wmPath, size_t regNum, size_t pattNum) {
    std::vector<Block> blocks = readImageAndSplitToBlocks(imagePath);
    std::vector<uchar> watermark = readCvzAndConvertToVector(wmPath);


    for (size_t i = 0; i < blocks.size(); ++i) {
        Population population(blocks[i], reg0Variations[regNum],  reg1Variations[regNum], maskWholeVector[pattNum], watermark[i % WM_SIZE]);
        population.GBO();
    }

    cv::Mat watermarkedImage = reconstructImageFromBlocks(blocks, 512, 512);
    const std::string& newImagePath = addPrefixToFilename(imagePath, "wm_");
    cv::imwrite(newImagePath, watermarkedImage);
}