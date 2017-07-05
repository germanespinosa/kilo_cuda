#ifndef KILO_CUDA
#define KILO_CUDA

#define TILELIMIT 256 
#define LINGRID (ROBOTS - 1) / TILELIMIT + 1
#define DIST(P1__, P2__) sqrt(pow(P1__.x - P2__.x,2) + pow(P1__.y - P2__.y,2))
#define COLLIDE(P1__, P2__) (sqrt(pow(P1__.x - P2__.x,2) + pow(P1__.y - P2__.y,2)) <= ROBOT_RADIUS)
#define INBOUNDS(P__) ((P__.x > ROBOT_RADIUS && P__.x < ARENA_WIDTH-ROBOT_RADIUS) && (P__.y > ROBOT_RADIUS && P__.y < ARENA_HEIGHT-ROBOT_RADIUS))
#define HRAND ((float)(curand(&robots[rid].hstate) % 1000)/1000)
#define SRAND ((float)(curand(&robots[rid].sstate) % 1000)/1000)
#define ROBOTID blockIdx.x*TILELIMIT+threadIdx.x;
#define SHAPEID blockIdx.y;
#include "robot.h"
void release_cuda_memory();
void initialize_shapes(Rectangle *rectangles);
void initialize_robots(Position *positions);
void compute_step();
void simulation_step();
Robot *download_robot_data();
void release_robots();
#endif