#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include <malloc.h>
#include <math.h>


#define SECONDS 600

#include "util.h"
#include "kilo_kernels.h"
#include "robot.h"

Position *init_positions ();

int main(int argc, char* argv[]) 
{
	Position *initial_positions=init_positions ();
	initialize_robots(initial_positions);
	free(initial_positions);
    for (int s=0; s<STEPS; s++)    
    {
        simulation_step();
    }
 	printf("done\n");     
}


Position *init_positions ()
{
    Position *positions=(Position *) malloc(sizeof(Position) * ROBOTS);
    unsigned int limitx = ARENA_WIDTH - 2 * ROBOT_RADIUS;
    unsigned int limity = ARENA_HEIGHT - 2 * ROBOT_RADIUS;
    for (unsigned int i=0; i<ROBOTS; i++)
    {
        bool collision = true;
        while(collision)
        {
            collision = false;
            positions[i].x = ROBOT_RADIUS + rand() % limitx;
            positions[i].y = ROBOT_RADIUS + rand() % limity;
            positions[i].theta = (2*PI/1000)*((float)(rand() % 1000));
            for (unsigned int j=0; j<i && !collision; j++)
            {
                if (DIST(positions[i],positions[j])<ROBOT_RADIUS)
                    collision=true;
            }
        }
    }
    return positions;
}