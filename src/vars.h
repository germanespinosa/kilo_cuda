//
// Created by jtebert on 6/26/17.
//

#ifndef PROJECT_VARS_H
#define PROJECT_VARS_H

#include <vector>
#include <string>
#include "util.h"

// General parameters
extern int timelimit;  // in seconds

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

// Display/UI parameters
extern bool showscene;
extern std::string ui_shapes_filename;
extern std::string ui_robots_filename;


#endif //PROJECT_VARS_H
