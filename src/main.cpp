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
	Position *initial_positions=init_positions();
	if (initial_positions == NULL)
	{
		printf("PROBLEMS ALLOCATING HOST MEMORY\n");
		return 1;
	}
	initialize_robots(initial_positions);

	Position *position = download_position_data();
	printf("CHECKING POSITION\n");
	for (int i = 0; i < ROBOTS; i++)
	{
		if (position[i].x != initial_positions[i].x || position[i].y != initial_positions[i].y || position[i].theta != initial_positions[i].theta)
		{
			printf("%d: %2.2f %2.2f - %2.2f %2.2f - %2.2f %2.2f\n", i, position[i].x, initial_positions[i].x , position[i].y ,initial_positions[i].y, position[i].theta, initial_positions[i].theta);
			printf("PROBLEMS MANAGING POSITION DATA IN/OUT GPU\n");
			return 1;
		}
	}

	Robot *robot = download_robot_data();
	printf("CHECKING ROBOT INITIALIZATION\n");
	for(int i = 0; i < ROBOTS; i++)
	{
		if (robot[i].position.x != initial_positions[i].x || robot[i].position.y != initial_positions[i].y || robot[i].position.theta != initial_positions[i].theta)
		{
			printf("%d: %2.2f %2.2f - %2.2f %2.2f - %2.2f %2.2f\n", i, robot[i].position.x, initial_positions[i].x , robot[i].position.y ,initial_positions[i].y, robot[i].position.theta, initial_positions[i].theta);
			printf("PROBLEMS MANAGING ROBOT DATA IN/OUT GPU\n");
			return 1;
		}
	}
	printf("DATA OK	\n");
	free(initial_positions);

	printf("RUNNING\n");
	for (int s=0; s<STEPS; s++)
    {
        simulation_step();
		if (s%100==0) printf ( "step:%d or %d\r" ,(s+1)/TICS_PER_SECOND, SECONDS);
    }
	printf("step:%d or %d\n", SECONDS, SECONDS);
	release_cuda_memory();
 	printf("\ndone\n");     
	getchar();
}

Position *init_positions ()
{
    Position *positions=(Position *) malloc(sizeof(Position) * ROBOTS);
	if (positions != NULL)
	{
		unsigned int limitx = ARENA_WIDTH - 2 * ROBOT_RADIUS;
		unsigned int limity = ARENA_HEIGHT - 2 * ROBOT_RADIUS;
		for (unsigned int i = 0; i < ROBOTS; i++)
		{
			bool collision = true;
			while (collision)
			{
				collision = false;
				positions[i].x = ROBOT_RADIUS + rand() % limitx;
				positions[i].y = ROBOT_RADIUS + rand() % limity;
				positions[i].theta = (2 * PI / 1000)*((float)(rand() % 1000));
				for (unsigned int j = 0; j < i && !collision; j++)
				{
					if (DIST(positions[i], positions[j]) < ROBOT_RADIUS)
						collision = true;
				}
			}
		}
	}
    return positions;
}