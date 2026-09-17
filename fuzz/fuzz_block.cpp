// GBO on a single 8x8 block: every pixel pattern, bit, attack type and scheme.
// The result must stay a valid block and be reproducible from the seed.

#include "fuzz_common.h"
#include "embedding_core.h"
#include "gbo.h"
#include "random_utils.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    fuzz::initLibrary();
    fuzz::Input in(data, size);
    fuzz::pickScheme(in);
    const uint64_t seed = in.take<uint64_t>();
    const uchar bit = in.take<uint8_t>() & 1;
    const AttackType attack = static_cast<AttackType>(in.take<uint8_t>() % 4);
    cv::Mat block(8, 8, CV_8UC1);
    in.fill(block.data, 64);
    const cv::Mat original = block.clone();

    seed_random_stream(seed, 0);
    GBO first(bit, block, attack);
    first.main_loop();
    FUZZ_CHECK(block.rows == 8 && block.cols == 8 && block.type() == CV_8UC1);

    cv::Mat again = original.clone();
    seed_random_stream(seed, 0);
    GBO second(bit, again, attack);
    second.main_loop();
    FUZZ_CHECK(cv::norm(block, again, cv::NORM_INF) == 0);
    return 0;
}
