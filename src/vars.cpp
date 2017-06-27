//
// Created by jtebert on 6/26/17.
//

#include "vars.h"

// General parameters
int timelimit = 180 * 60;  // seconds
bool showscene = true;

// Communication & dissemination

// Logging results

// Arena dimensions
float edge_width = 48;  // mm
int arena_width = 2400;  // mm
int arena_height = 2400;  // mm

// Arena parameters for shapes
std::vector<Rectangle> rects = {};

// Time constants for detection (exploration/observation + dissemination)