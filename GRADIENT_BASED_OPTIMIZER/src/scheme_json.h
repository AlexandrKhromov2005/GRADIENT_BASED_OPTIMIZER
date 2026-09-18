#ifndef SCHEME_JSON_H
#define SCHEME_JSON_H

#include <map>
#include <string>

#include "embedding_schemes.h"

// Parses the contents of embedding_schemes.json:
//
//   { "<scheme id>": { "name": "...", "description": "...",
//                      "REG0": [[row, col], ...], "REG1": [...], "ZONE0": [...] }, ... }
//
// The text must be one JSON object (RFC 8259, UTF-8; a leading byte order mark is skipped) with
// at least one scheme. In a scheme REG0, REG1 and ZONE0 are required, "name" (default: the id)
// and "description" are optional strings and other members are ignored. Ids, names and
// descriptions contain no control characters. Rows and columns are integers 0..7. A list is
// not empty and has no repeated position; REG0 and REG1 have no position in common; every
// ZONE0 position belongs to REG0 or REG1, the coefficients that make up S0 and S1.
//
// Returns false and describes the first problem in `error` ("line L, column C: ..."); `out`
// is assigned only on success.
bool parseSchemesJson(const std::string& text, std::map<std::string, EmbeddingScheme>& out,
                      std::string& error);

// Inputs above these limits are rejected.
constexpr size_t kSchemesJsonMaxBytes = 1 << 20;
constexpr size_t kSchemesJsonMaxSchemes = 256;
constexpr int kSchemesJsonMaxDepth = 32;  // nesting of ignored values

#endif // SCHEME_JSON_H
