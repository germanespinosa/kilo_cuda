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
//#include "kilobot.cpp"

Position *init_positions ();

int main(int argc, char* argv[]) {
	Position *initial_positions = init_positions ();
	initialize_robots(initial_positions);
	Robot *robot = download_robot_data();
	bool found_errors = false;
<<<<<<< HEAD
	const char *code_file = "C:\\Users\\germa\\Dropbox\\Research\\kilo_cuda\\src\\kilobot.cpp";
=======
>>>>>>> 30e0ae1723acd6948008a6ce9871206ef2fd7eab

	printf("CHECKING DATA\n");
	for (int i = 0; i < ROBOTS; i++) {
		if (robot[i].position.x != initial_positions[i].x || robot[i].position.y != initial_positions[i].y) {
			printf("%d: %2.2f %2.2f - %2.2f %2.2f - %2.2f %2.2f\n", i, robot[i].position.x, initial_positions[i].x , robot[i].position.y ,initial_positions[i].y, robot[i].position.theta, initial_positions[i].theta);
			found_errors = true;
		}
	}

	free(initial_positions);
	if (found_errors) {
		printf("PROBLEMS MANAGING DATA IN/OUT GPU\n");
		return 1;
	}
	printf("DATA OK	\n");
<<<<<<< HEAD

	printf("RUNNING\n");
	for (int s=0; s<STEPS; s++)
    {
=======
	for (int s=0; s<STEPS; s++) {
>>>>>>> 30e0ae1723acd6948008a6ce9871206ef2fd7eab
        simulation_step();
		if (s%100==0) printf ( "step:%d or %d\r\n" ,(s+1)/TICS_PER_SECOND, SECONDS);
    }
	printf("step:%d or %d\n", SECONDS, SECONDS);
<<<<<<< HEAD
	release_cuda_memory();
 	printf("\ndone\n");     
=======
	release_robots();
 	printf("\ndone\n");
>>>>>>> 30e0ae1723acd6948008a6ce9871206ef2fd7eab
	getchar();
    release_cuda_memory();
}

Position *init_positions () {
    Position *positions=(Position *) malloc(sizeof(Position) * ROBOTS);
    unsigned int limitx = ARENA_WIDTH - 2 * ROBOT_RADIUS;
    unsigned int limity = ARENA_HEIGHT - 2 * ROBOT_RADIUS;
    for (unsigned int i=0; i<ROBOTS; i++) {
        bool collision = true;
        while(collision) {
            collision = false;
            positions[i].x = ROBOT_RADIUS + rand() % limitx;
            positions[i].y = ROBOT_RADIUS + rand() % limity;
            positions[i].theta = (float)(2*PI/1000)*((float)(rand() % 1000));
            for (unsigned int j=0; j<i && !collision; j++) {
                if (DIST(positions[i], positions[j])<ROBOT_RADIUS)
                    collision=true;
            }
        }
    }
    return positions;
}