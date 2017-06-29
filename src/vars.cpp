//
// Created by jtebert on 6/26/17.
//

#include "vars.h"

// General parameters
int timelimit = 180 * 60;  // seconds

// Communication & dissemination

// Logging results

// Arena dimensions
float edge_width = 48;  // mm
int arena_width = 2400;  // mm
int arena_height = 2400;  // mm

// Arena parameters for shapes
std::vector<Rectangle> rects = {};

// Time constants for detection (exploration/observation + dissemination)

// Display/UI parameters
bool showscene = true;
std::string ui_shapes_filename = "../ui/shapes.tsv";
std::string ui_robots_filename = "../ui/robots.tsv";