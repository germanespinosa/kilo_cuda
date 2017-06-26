//
// Created by jtebert on 6/26/17.
//

#ifndef PROJECT_VARS_H
#define PROJECT_VARS_H

#include <vector>
#include "util.h"

// General parameters
extern int timelimit;  // in seconds
extern bool showscene;

// Communication & dissemination

// Logging results

// Arena dimensions
extern float edge_width;  // mm
extern int arena_width;  // mm
extern int arena_height;  // mm

// Arena parameters for shapes
extern std::vector<Rectangle> rects;

// Constants
const uint8_t TICS_PER_SECOND = 32;
#define PI 3.1416

// Time constants for detection (exploration/observation + dissemination)


#endif //PROJECT_VARS_H
