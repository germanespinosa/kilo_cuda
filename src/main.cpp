#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
//#include <cuda_runtime.h>
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
	Robot *robot = download_robot_data();
	/*
	for(int i = 0; i < ROBOTS; i++)
	{
		if (robot[i].position.x != initial_positions[i].x || robot[i].position.y != initial_positions[i].y)
		{
			printf("%d: %2.2f %2.2f - %2.2f %2.2f - %2.2f %2.2f\n", i, robot[i].position.x, initial_positions[i].x , robot[i].position.y ,initial_positions[i].y, robot[i].position.theta, initial_positions[i].theta);
		}
	}*/
	
	free(initial_positions);

    for (int s=0; s<STEPS; s++)    
    {
        simulation_step();
		if (s%10==0) printf ( "step:%d or %d\r" ,s+1,STEPS);
    }
	
 	printf("done\n");     
	getchar();
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