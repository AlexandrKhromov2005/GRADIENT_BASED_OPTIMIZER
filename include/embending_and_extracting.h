#ifndef EMBENDING_AND_EXTRACTING_H
#define EMBENDING_AND_EXTRACTING_H

#include <string>
#include <vector>
#include "process_image.h"
#include "smart_block.h"
#include "population.h"
#include "config.h"
#include "patterns.h"

void embend(const std::string& imagePath, const std::string& wmPath, size_t regNum, size_t pattNum);
void extract(const std::string& imagePath, const std::string& wmPath, size_t regNum, size_t pattNum);


#endif //EMBENDING_AND_EXTRACTING_H
