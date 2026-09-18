// Tests of the embedding_schemes.json parser and of loading schemes into the manager.
// Usage: scheme_json_test <path to the repository's embedding_schemes.json>

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "embedding_schemes.h"
#include "scheme_json.h"

namespace {

int failures = 0;

#define T_CHECK(cond)                                                        \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

using Schemes = std::map<std::string, EmbeddingScheme>;
using Coords = std::vector<std::pair<int, int>>;

const char* const kLists = "\"REG0\":[[0,1]],\"REG1\":[[1,0]],\"ZONE0\":[[0,1],[1,0]]";

std::string scheme(const std::string& members) { return "{\"s\":{" + members + "}}"; }

void accepts(const std::string& text) {
    Schemes out;
    std::string error;
    if (!parseSchemesJson(text, out, error)) {
        std::printf("FAILED: rejected (%s): %s\n", error.c_str(), text.c_str());
        ++failures;
    }
}

void rejects(const std::string& text, const std::string& expected_in_error) {
    Schemes out;
    out["untouched"] = EmbeddingScheme();
    std::string error;
    if (parseSchemesJson(text, out, error)) {
        std::printf("FAILED: accepted: %s\n", text.c_str());
        ++failures;
        return;
    }
    if (error.find(expected_in_error) == std::string::npos) {
        std::printf("FAILED: error \"%s\" does not mention \"%s\": %s\n", error.c_str(),
                    expected_in_error.c_str(), text.c_str());
        ++failures;
    }
    T_CHECK(out.size() == 1 && out.count("untouched") == 1);
}

void testValues() {
    Schemes out;
    std::string error;
    const std::string text =
        "\xEF\xBB\xBF \n{ \"id \\u00e9\\ud83d\\ude00\\\"\\\\\\/\" : {\n"
        "  \"description\" : \"d\", \"ignored \\b\\f\\n\\r\\t\" : [1, -2.5e+3, true, false, null, {\"a\": {\"b\": []}}, \"x\"],\n"
        "  \"REG0\" : [ [7 , 0] , [6,0] ], \"REG1\":[[0,7]], \"ZONE0\":[[0,7],[6,0],[7,0]] } }\r\n";
    T_CHECK(parseSchemesJson(text, out, error));
    T_CHECK(error.empty());
    T_CHECK(out.size() == 1);
    if (out.size() != 1) return;
    const std::string id = "id \xC3\xA9\xF0\x9F\x98\x80\"\\/";
    T_CHECK(out.begin()->first == id);
    const EmbeddingScheme& s = out.begin()->second;
    T_CHECK(s.name == id);  // no "name": the id is used
    T_CHECK(s.description == "d");
    T_CHECK((s.REG0 == Coords{{7, 0}, {6, 0}}));
    T_CHECK((s.REG1 == Coords{{0, 7}}));
    T_CHECK((s.ZONE0 == Coords{{0, 7}, {6, 0}, {7, 0}}));  // order is kept
}

void testAccepted() {
    accepts(scheme(kLists));
    accepts(scheme(std::string("\"name\":\"\",") + kLists));
    accepts("{\"a\":{" + std::string(kLists) + "},\"b\":{" + kLists + "}}");
    accepts(scheme(std::string(kLists) + ",\"x\":0,\"y\":-0,\"z\":0.0,\"w\":1E2"));

    // All 64 positions: the rows 0..3 in REG0, 4..7 in REG1.
    std::string half[2], all;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            const std::string pair = "[" + std::to_string(r) + "," + std::to_string(c) + "]";
            half[r / 4] += (half[r / 4].empty() ? "" : ",") + pair;
            all += (all.empty() ? "" : ",") + pair;
        }
    }
    accepts(scheme("\"REG0\":[" + half[0] + "],\"REG1\":[" + half[1] + "],\"ZONE0\":[" + all + "]"));
    accepts(scheme(std::string("\"name\":\"\xD0\x96 \xE2\x82\xAC \xF0\x9F\x98\x80\",") + kLists));  // raw UTF-8

    std::string deep = std::string(kSchemesJsonMaxDepth, '[') + std::string(kSchemesJsonMaxDepth, ']');
    accepts(scheme(std::string(kLists) + ",\"x\":" + deep));
}

void testRejected() {
    rejects("", "object of schemes");
    rejects("[]", "object of schemes");
    rejects("{}", "no schemes");
    rejects("{\"s\":1}", "must be an object");
    rejects("{\"\":{" + std::string(kLists) + "}}", "empty scheme id");
    rejects("{\"a\":{" + std::string(kLists) + "},\"a\":{" + kLists + "}}", "duplicate scheme id");
    rejects(scheme(kLists) + "x", "after the document");
    rejects(scheme(kLists) + std::string(1, '\0'), "after the document");
    rejects(scheme(std::string(kLists) + ","), "string expected");
    rejects("{\"s\":{" + std::string(kLists) + "}", "expected after a scheme");

    rejects(scheme("\"REG0\":[[0,1]],\"REG1\":[[1,0]]"), "needs REG0, REG1 and ZONE0");
    rejects(scheme(std::string(kLists) + ",\"REG0\":[[2,2]]"), "duplicate member");
    rejects(scheme("\"name\":1," + std::string(kLists)), "string expected");
    rejects(scheme("\"REG0\":[],\"REG1\":[[1,0]],\"ZONE0\":[[0,1]]"), "empty list");
    rejects(scheme("\"REG0\":[[0,1],[0,1]],\"REG1\":[[1,0]],\"ZONE0\":[[0,1]]"), "listed twice");
    rejects(scheme("\"REG0\":[[0,1]],\"REG1\":[[0,1]],\"ZONE0\":[[0,1]]"), "share a position");
    rejects(scheme("\"REG0\":[[0,1]],\"REG1\":[[1,0]],\"ZONE0\":[[0,1],[7,7]]"), "outside REG0 and REG1");

    const char* bad_index[] = {"8", "-1", "-0", "10", "07", "1.0", "1e0", "\"1\"", "null", "[1]", ""};
    for (const char* index : bad_index)
        rejects(scheme("\"REG0\":[[" + std::string(index) + ",1]],\"REG1\":[[1,0]],\"ZONE0\":[[0,1]]"), "integer 0..7");
    rejects(scheme("\"REG0\":[[0]],\"REG1\":[[1,0]],\"ZONE0\":[[0,1]]"), "between row and column");
    rejects(scheme("\"REG0\":[[0,1,2]],\"REG1\":[[1,0]],\"ZONE0\":[[0,1]]"), "after the column");
    rejects(scheme("\"REG0\":[0,1],\"REG1\":[[1,0]],\"ZONE0\":[[0,1]]"), "[row, col] expected");

    rejects(scheme("\"name\":\"a\nb\"," + std::string(kLists)), "control character");
    rejects(scheme("\"name\":\"\\x\"," + std::string(kLists)), "unknown escape");
    rejects(scheme("\"name\":\"\\u12g4\"," + std::string(kLists)), "hex digits");
    rejects(scheme("\"name\":\"\\ud800\"," + std::string(kLists)), "unpaired surrogate");
    rejects(scheme("\"name\":\"\\ud800\\u0041\"," + std::string(kLists)), "unpaired surrogate");
    rejects(scheme("\"name\":\"\\udc00\"," + std::string(kLists)), "unpaired surrogate");
    rejects(scheme("\"name\":\"abc"), "unterminated string");

    const char* bad_utf8[] = {"\x80", "\xC0\xAF", "\xC1\xBF", "\xC3", "\xC3\x28", "\xE0\x80\x80", "\xE2\x82",
                              "\xED\xA0\x80", "\xF0\x80\x80\x80", "\xF4\x90\x80\x80", "\xF5\x80\x80\x80", "\xFF"};
    for (const char* bytes : bad_utf8) {
        rejects(scheme("\"name\":\"" + std::string(bytes) + "\"," + kLists), "invalid UTF-8");
        rejects(scheme(std::string(kLists) + ",\"x\":\"" + bytes + "\""), "invalid UTF-8");
    }
    const char* controls[] = {"\\u0000", "\\u001b[31m", "\\n", "\\u007f", "\\u009b", "\xC2\x85"};
    for (const char* control : controls) {
        rejects(scheme("\"name\":\"a" + std::string(control) + "\"," + kLists), "control character in a scheme");
        rejects(scheme("\"description\":\"" + std::string(control) + "\"," + kLists), "control character in a scheme");
        rejects("{\"id" + std::string(control) + "\":{" + kLists + "}}", "control character in a scheme");
        accepts(scheme(std::string(kLists) + ",\"x\":\"" + control + "\""));  // ignored members are not printed
    }

    const char* bad_value[] = {"01", "-", "1.", ".5", "1e", "+1", "tru", "nul", "NaN", "'a'", "{\"a\"}", "[1,]", "{,}"};
    for (const char* value : bad_value) {
        Schemes out;
        std::string error;
        T_CHECK(!parseSchemesJson(scheme(std::string(kLists) + ",\"x\":" + value), out, error));
    }

    std::string deep = std::string(kSchemesJsonMaxDepth + 1, '[') + std::string(kSchemesJsonMaxDepth + 1, ']');
    rejects(scheme(std::string(kLists) + ",\"x\":" + deep), "nested too deeply");
    rejects(scheme(std::string(kLists) + ",\"x\":" + std::string(100000, '[')), "nested too deeply");

    std::string many = "{";
    for (size_t i = 0; i <= kSchemesJsonMaxSchemes; ++i)
        many += (i ? ",\"" : "\"") + std::to_string(i) + "\":{" + kLists + "}";
    rejects(many + "}", "too many schemes");
    rejects(scheme(kLists) + std::string(kSchemesJsonMaxBytes, ' '), "larger than");

    // Error position.
    Schemes out;
    std::string error;
    T_CHECK(!parseSchemesJson("{\n  \"s\": {\n    \"REG0\": [[9,1]]", out, error));
    T_CHECK(error.rfind("line 3, column 15:", 0) == 0);
}

void testRepositoryFile(const std::string& path) {
    auto& manager = EmbeddingSchemeManager::getInstance();
    T_CHECK(!manager.loadSchemes(path + ".missing"));
    T_CHECK(manager.loadSchemes(path));
    const std::vector<std::string> ids = {"extended_scheme", "scheme1", "scheme2", "scheme3", "standard_scheme"};
    T_CHECK(manager.getAvailableSchemes() == ids);

    const size_t zone_sizes[] = {25, 22, 22, 25, 22};
    for (size_t i = 0; i < ids.size(); ++i) {
        manager.setCurrentScheme(ids[i]);
        T_CHECK(CURRENT_VEC_SIZE == zone_sizes[i]);
        T_CHECK(getCurrentZONE0().size() == zone_sizes[i]);
        T_CHECK(getCurrentREG0().size() + getCurrentREG1().size() == zone_sizes[i]);
    }
    const EmbeddingScheme* scheme1 = manager.getScheme("scheme1");
    T_CHECK(scheme1 && scheme1->name == "Original Scheme");
    T_CHECK(scheme1 && scheme1->REG0.front() == std::make_pair(7, 1) && scheme1->ZONE0.back() == std::make_pair(1, 7));

    // A rejected text keeps the loaded set and the active scheme.
    manager.setCurrentScheme("scheme3");
    std::string error;
    T_CHECK(!manager.loadSchemesFromString("{\"s\":{}}", &error));
    T_CHECK(!error.empty());
    T_CHECK(manager.getAvailableSchemes() == ids);
    T_CHECK(CURRENT_VEC_SIZE == 25);

    // A new set without the active scheme selects its first scheme and that scheme's size.
    T_CHECK(manager.loadSchemesFromString("{\"b\":{" + std::string(kLists) + "},\"a\":{\"REG0\":[[0,1]],\"REG1\":[[1,0]],\"ZONE0\":[[1,0]]}}"));
    T_CHECK(manager.getCurrentScheme() == manager.getScheme("a"));
    T_CHECK(CURRENT_VEC_SIZE == 1);
    manager.setCurrentScheme("missing");  // unknown id: nothing changes
    T_CHECK(manager.getCurrentScheme() == manager.getScheme("a"));
    T_CHECK(CURRENT_VEC_SIZE == 1);

    // Reloading the active scheme with another ZONE0 updates the vector size.
    T_CHECK(manager.loadSchemesFromString("{\"a\":{" + std::string(kLists) + "}}"));
    T_CHECK(CURRENT_VEC_SIZE == 2);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::printf("usage: %s <embedding_schemes.json>\n", argv[0]);
        return 2;
    }
    testValues();
    testAccepted();
    testRejected();
    testRepositoryFile(argv[1]);
    if (failures == 0) std::printf("OK\n");
    else std::printf("%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
