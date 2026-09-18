// embedding_schemes.json parser: the input is the file contents.
// Rejected text must leave the loaded schemes alone; accepted text must give schemes that
// satisfy the documented rules, survive a write/parse round trip and work in the optimizer.

#include "fuzz_common.h"
#include "embedding_core.h"
#include "embedding_schemes.h"
#include "gbo.h"
#include "random_utils.h"
#include "scheme_json.h"

#include <algorithm>
#include <map>
#include <set>

namespace {

using Schemes = std::map<std::string, EmbeddingScheme>;
using Coords = std::vector<std::pair<int, int>>;

std::string quoted(const std::string& s) {
    static const char* hex = "0123456789abcdef";
    std::string out = "\"";
    for (const unsigned char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20) {
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else {
            out += static_cast<char>(c);
        }
    }
    return out + "\"";
}

std::string serialize(const Schemes& schemes) {
    std::string out = "{";
    bool first_scheme = true;
    for (const auto& item : schemes) {
        if (!first_scheme) out += ",";
        first_scheme = false;
        const EmbeddingScheme& scheme = item.second;
        out += quoted(item.first) + ":{\"name\":" + quoted(scheme.name) + ",\"description\":" + quoted(scheme.description);
        const std::pair<const char*, const Coords*> lists[] = {
            {"REG0", &scheme.REG0}, {"REG1", &scheme.REG1}, {"ZONE0", &scheme.ZONE0}};
        for (const auto& list : lists) {
            out += std::string(",\"") + list.first + "\":[";
            for (size_t i = 0; i < list.second->size(); ++i) {
                if (i > 0) out += ",";
                out += "[" + std::to_string((*list.second)[i].first) + "," + std::to_string((*list.second)[i].second) + "]";
            }
            out += "]";
        }
        out += "}";
    }
    return out + "}";
}

bool sameSchemes(const Schemes& a, const Schemes& b) {
    if (a.size() != b.size()) return false;
    for (auto i = a.begin(), j = b.begin(); i != a.end(); ++i, ++j) {
        if (i->first != j->first || i->second.name != j->second.name ||
            i->second.description != j->second.description || i->second.REG0 != j->second.REG0 ||
            i->second.REG1 != j->second.REG1 || i->second.ZONE0 != j->second.ZONE0)
            return false;
    }
    return true;
}

void checkList(const Coords& coords) {
    FUZZ_CHECK(!coords.empty() && coords.size() <= 64);
    std::set<std::pair<int, int>> seen;
    for (const auto& position : coords) {
        FUZZ_CHECK(position.first >= 0 && position.first <= 7 && position.second >= 0 && position.second <= 7);
        FUZZ_CHECK(seen.insert(position).second);
    }
}

uint64_t fnv1a(const uint8_t* data, size_t size) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) h = (h ^ data[i]) * 1099511628211ull;
    return h;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    fuzz::initLibrary();
    const std::string text(reinterpret_cast<const char*>(data), size);
    auto& manager = EmbeddingSchemeManager::getInstance();
    const std::vector<std::string> ids_before = manager.getAvailableSchemes();

    Schemes sentinel;
    sentinel["sentinel"] = EmbeddingScheme();
    Schemes parsed = sentinel;
    std::string error;
    const bool ok = parseSchemesJson(text, parsed, error);
    FUZZ_CHECK(manager.loadSchemesFromString(text) == ok);

    if (!ok) {
        FUZZ_CHECK(!error.empty());
        FUZZ_CHECK(sameSchemes(parsed, sentinel));
        FUZZ_CHECK(manager.getAvailableSchemes() == ids_before);
        return 0;
    }

    FUZZ_CHECK(error.empty());
    FUZZ_CHECK(size <= kSchemesJsonMaxBytes);
    FUZZ_CHECK(!parsed.empty() && parsed.size() <= kSchemesJsonMaxSchemes);
    for (const auto& item : parsed) {
        const EmbeddingScheme& scheme = item.second;
        FUZZ_CHECK(!item.first.empty());
        checkList(scheme.REG0);
        checkList(scheme.REG1);
        checkList(scheme.ZONE0);
        for (const auto& position : scheme.REG0)
            FUZZ_CHECK(std::find(scheme.REG1.begin(), scheme.REG1.end(), position) == scheme.REG1.end());
        for (const auto& position : scheme.ZONE0)
            FUZZ_CHECK(std::find(scheme.REG0.begin(), scheme.REG0.end(), position) != scheme.REG0.end() ||
                       std::find(scheme.REG1.begin(), scheme.REG1.end(), position) != scheme.REG1.end());
        for (const std::string* printed : {&item.first, &scheme.name, &scheme.description})
            for (const unsigned char c : *printed) FUZZ_CHECK(c >= 0x20 && c != 0x7F);
    }

    // A default name doubles the id, so the rewritten text may exceed the size limit.
    const std::string rewritten = serialize(parsed);
    if (rewritten.size() <= kSchemesJsonMaxBytes) {
        Schemes reparsed;
        FUZZ_CHECK(parseSchemesJson(rewritten, reparsed, error));
        FUZZ_CHECK(sameSchemes(parsed, reparsed));
    }

    // The loaded schemes are what the embedding code now runs on.
    uint64_t h = fnv1a(data, size);
    auto chosen = parsed.begin();
    std::advance(chosen, static_cast<long>(h % parsed.size()));
    manager.setCurrentScheme(chosen->first);
    FUZZ_CHECK(manager.getCurrentScheme() != nullptr);
    FUZZ_CHECK(CURRENT_VEC_SIZE == chosen->second.ZONE0.size());

    cv::Mat block(8, 8, CV_8UC1);
    for (int i = 0; i < 64; ++i) {
        h = h * 6364136223846793005ull + 1442695040888963407ull;
        block.data[i] = static_cast<uint8_t>(h >> 56);
    }
    const int bit = extractBitFromBlock(block, true);
    FUZZ_CHECK(bit == 0 || bit == 1);

    if ((h >> 20) % 8 == 0) {  // the optimizer is slow: one accepted input in eight
        const cv::Mat original = block.clone();
        seed_random_stream(h, 0);
        GBO first(static_cast<uchar>(h & 1), block, static_cast<AttackType>((h >> 8) % 4));
        first.main_loop();
        FUZZ_CHECK(block.rows == 8 && block.cols == 8 && block.type() == CV_8UC1);

        cv::Mat again = original.clone();
        seed_random_stream(h, 0);
        GBO second(static_cast<uchar>(h & 1), again, static_cast<AttackType>((h >> 8) % 4));
        second.main_loop();
        FUZZ_CHECK(cv::norm(block, again, cv::NORM_INF) == 0);
    }
    return 0;
}
