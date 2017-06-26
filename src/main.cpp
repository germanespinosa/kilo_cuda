#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include <malloc.h>
#include <math.h>
#include <vector>
#include "vars.h"
#include "util.h"


#define SECONDS 600

#include "util.h"
#include "kilo_kernels.h"
#include "robot.h"

Position *init_positions ();
Robot *robots_here;

int main(int argc, char* argv[]) 
{
	Position *initial_positions=init_positions();

    // Sanity check: print positions
    printf("Positions from init_positions():\n");
    for (unsigned int i=0; i<ROBOTS; i++) std::cout << "(" << initial_positions[i].x << ", " << initial_positions[i].y << ") " << initial_positions[i].theta << std::endl;


	initialize_robots(initial_positions);
	free(initial_positions);
    for (int s=0; s<timelimit*TICS_PER_SECOND; s++) {
        // Update robots in GPU
        simulation_step();
        // Download updated robots from GPU -> CPU and get pointer to robot array
        robots_here = download_robot_data();

        // Sanity check: print position of 1st robot
        if (s == 0) {
            printf("\nPositions from kilo_kernels initialize_robot_data_kernel()\n");
            for (unsigned int i=0; i<ROBOTS; i++) std::cout << "(" << robots_here[i].position.x << ", " << robots_here[i].position.y << ") " << initial_positions[i].theta << std::endl;
        }
        //if (s % 100 == 0) printf("step: %d of %d\r\n", s+1, timelimit*TICS_PER_SECOND);
    }
 	printf("done\n");     
}


Position *init_positions ()
{
    Position *positions=(Position *) malloc(sizeof(Position) * ROBOTS);
    unsigned int limitx = ARENA_WIDTH - 2 * ROBOT_RADIUS;
    unsigned int limity = ARENA_HEIGHT - 2 * ROBOT_RADIUS;
    srand(time(NULL));
    for (unsigned int i=0; i<ROBOTS; i++)
    {
        bool collision = true;
        while(collision)
        {
            collision = false;
            positions[i].x = (float)ROBOT_RADIUS + rand() % limitx;
            positions[i].y = (float)ROBOT_RADIUS + rand() % limity;
            positions[i].theta = (float)(2*PI/1000)*((float)(rand() % 1000));
            for (unsigned int j=0; j<i && !collision; j++)
            {
                if (DIST(positions[i],positions[j])<ROBOT_RADIUS)
                    collision=true;
            }
        }
    }
    return positions;
}
