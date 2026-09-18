#include "scheme_json.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace {

using Coords = std::vector<std::pair<int, int>>;

struct ParseError {
    size_t pos;
    std::string message;
};

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    std::map<std::string, EmbeddingScheme> parseDocument() {
        std::map<std::string, EmbeddingScheme> schemes;
        if (s_.compare(0, 3, "\xEF\xBB\xBF") == 0) pos_ = 3;  // UTF-8 byte order mark
        skipSpace();
        expect('{', "the document must be an object of schemes");
        skipSpace();
        if (!consume('}')) {
            do {
                skipSpace();
                const size_t id_pos = pos_;
                const std::string id = parseString();
                if (id.empty()) fail(id_pos, "empty scheme id");
                requirePrintable(id, id_pos);
                if (schemes.count(id)) fail(id_pos, "duplicate scheme id \"" + id + "\"");
                if (schemes.size() == kSchemesJsonMaxSchemes) fail(id_pos, "too many schemes");
                skipSpace();
                expect(':', "':' expected after the scheme id");
                skipSpace();
                schemes[id] = parseScheme(id);
                skipSpace();
            } while (consume(','));
            expect('}', "',' or '}' expected after a scheme");
        }
        skipSpace();
        if (pos_ != s_.size()) fail(pos_, "unexpected text after the document");
        if (schemes.empty()) fail(0, "no schemes defined");
        return schemes;
    }

private:
    [[noreturn]] void fail(size_t pos, const std::string& message) const { throw ParseError{pos, message}; }

    bool atEnd() const { return pos_ >= s_.size(); }
    char peek() const { return atEnd() ? '\0' : s_[pos_]; }

    void skipSpace() {
        while (!atEnd() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) ++pos_;
    }

    bool consume(char c) {
        if (atEnd() || s_[pos_] != c) return false;
        ++pos_;
        return true;
    }

    void expect(char c, const char* message) {
        if (!consume(c)) fail(pos_, message);
    }

    EmbeddingScheme parseScheme(const std::string& id) {
        const size_t scheme_pos = pos_;
        expect('{', "a scheme must be an object");
        EmbeddingScheme scheme;
        bool has_name = false, has_description = false, has_reg0 = false, has_reg1 = false, has_zone0 = false;
        skipSpace();
        if (!consume('}')) {
            do {
                skipSpace();
                const size_t key_pos = pos_;
                const std::string key = parseString();
                skipSpace();
                expect(':', "':' expected after the member name");
                skipSpace();
                if (key == "name") {
                    readOnce(has_name, key_pos, key);
                    const size_t value_pos = pos_;
                    scheme.name = parseString();
                    requirePrintable(scheme.name, value_pos);
                } else if (key == "description") {
                    readOnce(has_description, key_pos, key);
                    const size_t value_pos = pos_;
                    scheme.description = parseString();
                    requirePrintable(scheme.description, value_pos);
                } else if (key == "REG0") {
                    readOnce(has_reg0, key_pos, key);
                    scheme.REG0 = parseCoords();
                } else if (key == "REG1") {
                    readOnce(has_reg1, key_pos, key);
                    scheme.REG1 = parseCoords();
                } else if (key == "ZONE0") {
                    readOnce(has_zone0, key_pos, key);
                    scheme.ZONE0 = parseCoords();
                } else {
                    skipValue(0);
                }
                skipSpace();
            } while (consume(','));
            expect('}', "',' or '}' expected after a scheme member");
        }
        if (!has_reg0 || !has_reg1 || !has_zone0) fail(scheme_pos, "scheme \"" + id + "\" needs REG0, REG1 and ZONE0");
        for (const auto& position : scheme.REG0) {
            if (std::find(scheme.REG1.begin(), scheme.REG1.end(), position) != scheme.REG1.end())
                fail(scheme_pos, "scheme \"" + id + "\": REG0 and REG1 share a position");
        }
        for (const auto& position : scheme.ZONE0) {
            if (std::find(scheme.REG0.begin(), scheme.REG0.end(), position) == scheme.REG0.end() &&
                std::find(scheme.REG1.begin(), scheme.REG1.end(), position) == scheme.REG1.end())
                fail(scheme_pos, "scheme \"" + id + "\": ZONE0 has a position outside REG0 and REG1");
        }
        if (!has_name) scheme.name = id;
        return scheme;
    }

    void readOnce(bool& seen, size_t key_pos, const std::string& key) {
        if (seen) fail(key_pos, "duplicate member \"" + key + "\"");
        seen = true;
    }

    // [[row, col], ...]: not empty, every position inside the 8x8 block and listed once.
    Coords parseCoords() {
        const size_t list_pos = pos_;
        expect('[', "a list of [row, col] pairs expected");
        Coords coords;
        skipSpace();
        if (!consume(']')) {
            do {
                skipSpace();
                const size_t pair_pos = pos_;
                expect('[', "[row, col] expected");
                skipSpace();
                const int row = parseIndex();
                skipSpace();
                expect(',', "',' expected between row and column");
                skipSpace();
                const int col = parseIndex();
                skipSpace();
                expect(']', "']' expected after the column");
                const std::pair<int, int> position(row, col);
                if (std::find(coords.begin(), coords.end(), position) != coords.end())
                    fail(pair_pos, "position listed twice");
                coords.push_back(position);  // at most 64 distinct positions exist
                skipSpace();
            } while (consume(','));
            expect(']', "',' or ']' expected after a pair");
        }
        if (coords.empty()) fail(list_pos, "empty list of positions");
        return coords;
    }

    // A row or column: one digit 0..7, no sign, fraction or exponent.
    int parseIndex() {
        const char c = peek();
        if (c < '0' || c > '9') fail(pos_, "integer 0..7 expected");
        const size_t start = pos_++;
        const char next = peek();
        if ((next >= '0' && next <= '9') || next == '.' || next == 'e' || next == 'E' || c > '7')
            fail(start, "integer 0..7 expected");
        return c - '0';
    }

    std::string parseString() {
        expect('"', "string expected");
        std::string out;
        for (;;) {
            if (atEnd()) fail(pos_, "unterminated string");
            const unsigned char c = static_cast<unsigned char>(s_[pos_++]);
            if (c == '"') return out;
            if (c < 0x20) fail(pos_ - 1, "control character in a string");
            if (c >= 0x80) {
                --pos_;
                copyUtf8Sequence(out);
                continue;
            }
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (atEnd()) fail(pos_, "unterminated string");
            const char escape = s_[pos_++];
            switch (escape) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': appendUtf8(out, parseCodePoint()); break;
                default: fail(pos_ - 1, "unknown escape sequence");
            }
        }
    }

    // Copies one multi-byte UTF-8 sequence; overlong forms, surrogates and values above
    // U+10FFFF are errors.
    void copyUtf8Sequence(std::string& out) {
        const size_t start = pos_;
        const unsigned char lead = static_cast<unsigned char>(s_[pos_]);
        const int length = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : 2;
        if (lead < 0xC2 || lead > 0xF4 || s_.size() - pos_ < static_cast<size_t>(length)) fail(start, "invalid UTF-8");
        unsigned cp = lead & (0x3Fu >> (length - 1));
        for (int i = 1; i < length; ++i) {
            const unsigned char next = static_cast<unsigned char>(s_[pos_ + i]);
            if ((next & 0xC0) != 0x80) fail(start, "invalid UTF-8");
            cp = (cp << 6) | (next & 0x3Fu);
        }
        const unsigned smallest[] = {0, 0, 0x80, 0x800, 0x10000};
        if (cp < smallest[length] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) fail(start, "invalid UTF-8");
        out.append(s_, pos_, static_cast<size_t>(length));
        pos_ += length;
    }

    // Ids, names and descriptions are printed to the terminal: no C0/C1 controls or DEL.
    void requirePrintable(const std::string& value, size_t pos) const {
        for (size_t i = 0; i < value.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(value[i]);
            const bool c1 = c == 0xC2 && i + 1 < value.size() && static_cast<unsigned char>(value[i + 1]) < 0xA0;
            if (c < 0x20 || c == 0x7F || c1) fail(pos, "control character in a scheme id, name or description");
        }
    }

    unsigned parseHex4() {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = peek();
            unsigned digit;
            if (c >= '0' && c <= '9') digit = static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') digit = static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') digit = static_cast<unsigned>(c - 'A' + 10);
            else fail(pos_, "four hex digits expected after \\u");
            value = value * 16 + digit;
            ++pos_;
        }
        return value;
    }

    // Called after "\u"; joins a surrogate pair into one code point.
    unsigned parseCodePoint() {
        const size_t start = pos_;
        const unsigned first = parseHex4();
        if (first >= 0xDC00 && first <= 0xDFFF) fail(start, "unpaired surrogate");
        if (first < 0xD800 || first > 0xDBFF) return first;
        if (!consume('\\') || !consume('u')) fail(start, "unpaired surrogate");
        const unsigned second = parseHex4();
        if (second < 0xDC00 || second > 0xDFFF) fail(start, "unpaired surrogate");
        return 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
    }

    static void appendUtf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    // Any JSON value, checked for syntax and dropped.
    void skipValue(int depth) {
        if (depth >= kSchemesJsonMaxDepth) fail(pos_, "value nested too deeply");
        const char c = peek();
        if (c == '"') {
            parseString();
        } else if (c == '{') {
            ++pos_;
            skipSpace();
            if (consume('}')) return;
            do {
                skipSpace();
                parseString();
                skipSpace();
                expect(':', "':' expected after the member name");
                skipSpace();
                skipValue(depth + 1);
                skipSpace();
            } while (consume(','));
            expect('}', "',' or '}' expected");
        } else if (c == '[') {
            ++pos_;
            skipSpace();
            if (consume(']')) return;
            do {
                skipSpace();
                skipValue(depth + 1);
                skipSpace();
            } while (consume(','));
            expect(']', "',' or ']' expected");
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            skipNumber();
        } else if (!skipLiteral("true") && !skipLiteral("false") && !skipLiteral("null")) {
            fail(pos_, "value expected");
        }
    }

    bool skipLiteral(const char* word) {
        const size_t length = std::char_traits<char>::length(word);
        if (s_.compare(pos_, length, word) != 0) return false;
        pos_ += length;
        return true;
    }

    size_t skipDigits() {
        const size_t start = pos_;
        while (peek() >= '0' && peek() <= '9') ++pos_;
        return pos_ - start;
    }

    void skipNumber() {
        const size_t start = pos_;
        consume('-');
        if (!consume('0') && skipDigits() == 0) fail(start, "malformed number");
        if (consume('.') && skipDigits() == 0) fail(start, "malformed number");
        if (consume('e') || consume('E')) {
            if (!consume('+')) consume('-');
            if (skipDigits() == 0) fail(start, "malformed number");
        }
    }

    const std::string& s_;
    size_t pos_ = 0;
};

std::string describe(const std::string& text, const ParseError& e) {
    size_t line = 1, column = 1;
    for (size_t i = 0; i < e.pos && i < text.size(); ++i) {
        if (text[i] == '\n') {
            ++line;
            column = 1;
        } else {
            ++column;
        }
    }
    return "line " + std::to_string(line) + ", column " + std::to_string(column) + ": " + e.message;
}

} // namespace

bool parseSchemesJson(const std::string& text, std::map<std::string, EmbeddingScheme>& out,
                      std::string& error) {
    if (text.size() > kSchemesJsonMaxBytes) {
        error = "file is larger than " + std::to_string(kSchemesJsonMaxBytes) + " bytes";
        return false;
    }
    try {
        out = Parser(text).parseDocument();
    } catch (const ParseError& e) {
        error = describe(text, e);
        return false;
    }
    error.clear();
    return true;
}
