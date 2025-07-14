#include <atomic>
#include "../include/smart_block.h"

Block::Block(cv::Mat_<double> sd) : spatialDomain_(sd) {
    cv::dct(sd, frequencyDomain_);
    frequencyDomain = frequencyDomain_;
}

//psnr is for calculating PSNR between spatialDomain_ and spatialDomain
double Block::psnr() {
        double mse = 0.0; 

        for (int i = 0; i < spatialDomain_.rows; ++i) {
            for (int j = 0; j < spatialDomain_.cols; ++j) {
                double diff = spatialDomain_(i, j) - spatialDomain(i, j);
                mse += diff * diff;
            }
        }
        mse /= (spatialDomain_.rows * spatialDomain_.cols);

        if (mse <= 1e-10) {
            return 100; //in case simillar blocks
        }

        double psnr = 10.0 * log10(65025.0 / mse);
        return psnr;
}

//calcAbsSum is for calculating S0 and S1
double Block::calcAbsSum(const Mask& mask) {
        std::atomic<double> totalSum{0.0};

    cv::parallel_for_(cv::Range(0, mask.size()), [&](const cv::Range& range) {
        double localSum = 0.0;
        for (int i = range.start; i < range.end; ++i) {
            const auto& [y, x] = mask[i];
            localSum += std::fabs(frequencyDomain(y, x));
        }
        totalSum.fetch_add(localSum, std::memory_order_relaxed);
    });

    return totalSum;
}