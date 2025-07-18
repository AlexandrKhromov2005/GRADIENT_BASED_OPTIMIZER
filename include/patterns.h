#ifndef PATTERNS_H
#define PATTERNS_H

#include <vector>
#include <utility> 
#include <cstddef>
#include "types.h"

const MaskCollection reg0Variations = {
    { {7, 1}, {6, 1}, {5, 1}, {5, 3}, {4, 3}, {3, 3}, {3, 5}, {2, 5}, {1, 5}, {1, 7}, {0, 7} },
    { {7, 0}, {7, 1}, {6, 0}, {6, 1}, {5, 1}, {4, 4}, {3, 4}, {3, 5}, {2, 4}, {2, 5}, {1, 5} },
    { {6, 0}, {7, 1}, {5, 1}, {6, 2}, {4, 3}, {3, 5}, {2, 4}, {2, 6}, {1, 5}, {1, 7}, {0, 6} },
    { {6, 0}, {5, 1}, {4, 2}, {3, 3}, {3, 4}, {2, 4}, {2, 5}, {1, 5}, {1, 6}, {0, 6}, {0, 7} },
    { {7, 0}, {6, 0}, {5, 1}, {5, 2}, {4, 2}, {3, 3}, {3, 4}, {2, 4}, {1, 5}, {1, 6}, {0, 6} },
    { {7, 1}, {6, 0}, {5, 2}, {5, 3}, {4, 2}, {3, 5}, {2, 4}, {2, 5}, {1, 7}, {0, 6}, {0, 7} },
    { {7, 0}, {7, 1}, {6, 0}, {6, 1}, {6, 2}, {5, 1}, {5, 2}, {1, 6}, {1, 7}, {0, 6}, {0, 7} },
    { {7, 0}, {7, 1}, {5, 1}, {5, 2}, {5, 3}, {3, 3}, {3, 4}, {3, 5}, {1, 5}, {1, 6}, {1, 7} },
    { {6, 0}, {6, 1}, {5, 1}, {5, 2}, {5, 3}, {2, 4}, {2, 5}, {2, 6}, {1, 5}, {1, 6}, {1, 7} },
    { {7, 0}, {7, 1}, {6, 0}, {6, 1}, {6, 2}, {5, 1}, {5, 2}, {4, 2}, {4, 3}, {3, 3}, {2, 4} },
    { {6, 1}, {6, 2}, {5, 2}, {5, 3}, {4, 3}, {4, 4}, {3, 4}, {2, 4}, {2, 5}, {1, 5}, {1, 6} },
    { {7, 0}, {6, 0}, {6, 1}, {5, 1}, {5, 2}, {4, 3}, {3, 3}, {3, 5}, {2, 4}, {1, 5}, {0, 6} },
    { {7, 0}, {5, 2}, {5, 3}, {4, 2}, {4, 3}, {4, 4}, {3, 3}, {3, 4}, {1, 6}, {1, 7}, {0, 6} },
    { {7, 0}, {7, 1}, {6, 0}, {6, 1}, {6, 2}, {5, 1}, {5, 3}, {4, 4}, {3, 4}, {3, 5}, {2, 5} },
    { {7, 0}, {7, 1}, {6, 0}, {6, 1}, {6, 2}, {5, 1}, {5, 2}, {5, 3}, {4, 2}, {4, 3}, {3, 3} },
    { {7, 0}, {7, 1}, {6, 0}, {6, 1}, {5, 3}, {4, 2}, {4, 4}, {3, 3}, {2, 5}, {2, 6}, {1, 5} },
    { {7, 0}, {7, 1}, {6, 0}, {6, 2}, {5, 1}, {5, 3}, {4, 2}, {4, 4}, {3, 4}, {2, 5}, {1, 6} },
    { {6, 1}, {6, 2}, {5, 1}, {5, 2}, {3, 4}, {3, 5}, {2, 4}, {2, 5}, {2, 6}, {1, 5}, {1, 6} },
    { {6, 2}, {5, 2}, {5, 3}, {4, 2}, {4, 3}, {3, 3}, {3, 4}, {3, 5}, {2, 4}, {2, 5}, {2, 6} }
};

const MaskCollection reg1Variations = {
    { {7, 0}, {6, 0}, {6, 2}, {5, 2}, {4, 2}, {4, 4}, {3, 4}, {2, 4}, {2, 6}, {1, 6}, {0, 6} },
    { {6, 2}, {5, 2}, {5, 3}, {4, 2}, {4, 3}, {3, 3}, {2, 6}, {1, 6}, {1, 7}, {0, 6}, {0, 7} },
    { {7, 0}, {6, 1}, {5, 2}, {5, 3}, {4, 2}, {4, 4}, {3, 3}, {3, 4}, {2, 5}, {1, 6}, {0, 7} },
    { {7, 0}, {7, 1}, {6, 1}, {6, 2}, {5, 2}, {5, 3}, {4, 3}, {4, 4}, {3, 5}, {2, 6}, {1, 7} },
    { {7, 1}, {6, 1}, {6, 2}, {5, 3}, {4, 3}, {4, 4}, {3, 5}, {2, 5}, {2, 6}, {1, 7}, {0, 7} },
    { {7, 0}, {6, 1}, {6, 2}, {5, 1}, {4, 3}, {4, 4}, {3, 3}, {3, 4}, {2, 6}, {1, 5}, {1, 6} },
    { {5, 3}, {4, 2}, {4, 3}, {4, 4}, {3, 3}, {3, 4}, {3, 5}, {2, 4}, {2, 5}, {2, 6}, {1, 5} },
    { {6, 0}, {6, 1}, {6, 2}, {4, 2}, {4, 3}, {4, 4}, {2, 4}, {2, 5}, {2, 6}, {0, 6}, {0, 7} },
    { {7, 0}, {7, 1}, {6, 2}, {4, 2}, {4, 3}, {4, 4}, {3, 4}, {3, 5}, {3, 3}, {0, 6}, {0, 7} },
    { {5, 3}, {4, 4}, {3, 4}, {3, 5}, {2, 5}, {2, 6}, {1, 5}, {1, 6}, {1, 7}, {0, 6}, {0, 7} },
    { {7, 0}, {7, 1}, {6, 0}, {5, 1}, {4, 2}, {3, 3}, {3, 5}, {2, 6}, {1, 7}, {0, 6}, {0, 7} },
    { {7, 1}, {6, 2}, {5, 3}, {4, 2}, {4, 4}, {3, 4}, {2, 5}, {2, 6}, {1, 6}, {1, 7}, {0, 7} },
    { {7, 1}, {6, 0}, {6, 1}, {6, 2}, {5, 1}, {3, 5}, {2, 4}, {2, 5}, {2, 6}, {1, 5}, {0, 7} },
    { {5, 2}, {4, 2}, {4, 3}, {3, 3}, {2, 4}, {2, 6}, {1, 5}, {1, 6}, {1, 7}, {0, 6}, {0, 7} },
    { {4, 4}, {3, 4}, {3, 5}, {2, 4}, {2, 5}, {2, 6}, {1, 5}, {1, 6}, {1, 7}, {0, 6}, {0, 7} },
    { {6, 2}, {5, 1}, {5, 2}, {4, 3}, {3, 4}, {3, 5}, {2, 4}, {1, 6}, {1, 7}, {0, 6}, {0, 7} },
    { {6, 1}, {5, 2}, {4, 3}, {3, 3}, {3, 5}, {2, 4}, {2, 6}, {1, 5}, {1, 7}, {0, 6}, {0, 7} },
    { {7, 0}, {7, 1}, {6, 0}, {5, 3}, {4, 2}, {4, 3}, {4, 4}, {3, 3}, {1, 7}, {0, 6}, {0, 7} },
    { {7, 0}, {7, 1}, {6, 0}, {6, 1}, {5, 1}, {4, 4}, {1, 5}, {1, 6}, {1, 7}, {0, 6}, {0, 7} }
};

const MaskCollection maskWholeVector = {
    { {6, 0}, {5, 1}, {4, 2}, {3, 3}, {2, 4}, {1, 5}, {0, 6}, {0, 7}, {1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}, {7, 0}, {7, 1}, {6, 2}, {5, 3}, {4, 4}, {3, 5}, {2, 6}, {1, 7} },
};

#endif // PATTERNS_H