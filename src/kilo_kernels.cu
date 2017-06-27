#include "kilo_kernels.h"
#include <curand.h>
#include <curand_kernel.h>

Robot      *cuda_robots;
Position   *cuda_next_positions;
Rectangle  *cuda_light_shapes;

Robot local_robots[ROBOTS];


__global__ void compute_step(Robot *robots, Position *next_position);
__global__ void collision_and_comms(Robot *robots,Position *next_position);
__global__ void update_state(Robot *robots,Position *next_position);
__global__ void initialize_robot_data_kernel(Robot *robots, Position *positions);
__global__ void commpute_step_kernel(Robot *robots,Step *step);
__global__ void compute_light(Robot *robots, Rectangle *cuda_light_shapes);

static dim3 lingrid(1,1);
static dim3 cuadgrid(ROBOTS,1);
static dim3 block(ROBOTS,1);
static dim3 shapesgrid;

void initialize_shapes(Rectangle *rectangles, int shapecount)
{
    // Upload shapes to GPU memory (light/shapes are static; occurs once at initialization)
    cudaMalloc((void**)&cuda_light_shapes, sizeof(Rectangle) * shapecount);
	dim3 grid(shapecount,1);
	shapesgrid = grid;
 	cudaMemcpy(cuda_light_shapes, rectangles, sizeof(Rectangle) * shapecount, cudaMemcpyHostToDevice);
}

void initialize_robots(Position *positions)
{
    // Upload initial robots/positions to GPU memory
    cudaMalloc((void**)&cuda_robots, sizeof(Robot) * ROBOTS);
	cudaMalloc((void**)&cuda_next_positions, sizeof(Position) * ROBOTS);
	cudaMemcpy(cuda_next_positions, positions, sizeof(Position) * ROBOTS, cudaMemcpyHostToDevice);
	initialize_robot_data_kernel <<< lingrid, block >>> ( cuda_robots, cuda_next_positions );
}

void simulation_step()
{
    // Compute next positions/communications
	compute_step <<< lingrid, block >>> ( cuda_robots, cuda_next_positions );
    // Compute light sensor values from shapes
	compute_light <<< shapesgrid, block >>> ( cuda_robots, cuda_light_shapes );
    // Check if communcations/movements valid
	collision_and_comms <<< cuadgrid, block >>> ( cuda_robots, cuda_next_positions );
    // Update next state from validity checks
	update_state <<< lingrid, block >>> ( cuda_robots, cuda_next_positions );
	// Download robot data to CPU
	cudaMemcpy(local_robots, cuda_robots, sizeof(Robot) * ROBOTS, cudaMemcpyDeviceToHost);
	// Repopulate robots state
	for (int rid=0;rid<ROBOTS;rid++)
	{
		// execute controller loop
	}
	cudaMemcpy(cuda_robots, local_robots, sizeof(Robot) * ROBOTS, cudaMemcpyHostToDevice);
	// upload state changes
	
}

Robot *download_robot_data()
{
 	cudaMemcpy(local_robots, cuda_robots, sizeof(Robot) * ROBOTS, cudaMemcpyDeviceToHost);
    return local_robots;
}

__global__ void compute_light(Robot *robots, Rectangle *cuda_light_shapes)
{
    // Calculate light sensor values from rectangles
    unsigned int sid = blockIdx.x;
	unsigned int rid = threadIdx.x;
    // Check if robot is in shape
    // If it is, set light to 1000
    // TODO: How to check/reset light to 0 at each time step (essentially want 1 output ["any"/"or"] from all combined)
    // TODO: How to deal with border area (gray light; maybe have to convert the way this is checked?)
}

__global__ void compute_step(Robot *robots, Position *next_position)
{
    unsigned int rid = threadIdx.x;
	//compute the movement needed	
	Step step;
	if (robots[rid].left_motor == 0) robots[rid].left_motor_active = false;
	if (robots[rid].right_motor == 0) robots[rid].right_motor_active = false;

	float turn_error  = HRAND * robots[rid].movement.turn_error - robots[rid].movement.turn_error / 2;
	float speed_error = HRAND * robots[rid].movement.speed_error - robots[rid].movement.speed_error / 2;
	
	step.turn = robots[rid].left_motor_active ? (robots[rid].right_motor_active ? robots[rid].movement.turn_forward: robots[rid].movement.turn_left) : (robots[rid].right_motor_active ? robots[rid].movement.turn_right: 0);
	step.speed = robots[rid].left_motor_active ? (robots[rid].right_motor_active ? robots[rid].movement.speed_forward: robots[rid].movement.speed_left) : (robots[rid].right_motor_active ? robots[rid].movement.speed_right: 0);
	
	step.turn += step.turn ? turn_error : 0 ;
	step.speed += step.speed ? speed_error : 0 ;
	
	
	//compute the next position	
    robots[rid].position.theta+=step.turn;
	Position temp_p;
    temp_p.theta = robots[rid].position.theta;
    temp_p.x = robots[rid].position.x + cos(robots[rid].position.theta) * step.speed;
    temp_p.y = robots[rid].position.y + sin(robots[rid].position.theta) * step.speed;
    if (INBOUNDS(temp_p))
    {
		next_position[rid]=temp_p;
	}
	else
    {
		next_position[rid]=robots[rid].position;
	}
}

__global__ void collision_and_comms(Robot *robots, Position *next_position)
{
    unsigned int nid = blockIdx.x;
    unsigned int rid = threadIdx.x;
	float d = DIST(robots[rid].position, next_position[nid]);
	float range_error = robots[rid].comm.range_error*HRAND - robots[rid].comm.range_error / 2;
	float range=robots[rid].comm.range + range_error;
	
	if (robots[nid].tx_flag && HRAND>robots[rid].comm.range_error && d<range)
	{
		robots[nid].rx_flag=true;
		for (int i=0;i<MESSAGE_SIZE;i++)
		{
			robots[nid].message_rx[i] = robots[rid].message_rx[i];
		}
	}
	
	if (d<ROBOT_RADIUS)
	{
		next_position[nid]=robots[nid].position;
	}
}

__global__ void update_state(Robot *robots,Position *next_position)
{
    unsigned int rid = threadIdx.x;
	robots[rid].position=next_position[rid];
	robots[rid].tx_flag=false;
}

__global__ void initialize_robot_data_kernel(Robot *robots, Position *positions)
{
    unsigned int rid = threadIdx.x;
	// initialize random seeds
	// all robots have the same soft random sequence
	curand_init(0, 0, 0, &robots[rid].sstate);
	// all robots have different hard random sequence
	curand_init(rid, 0, 0, &robots[rid].hstate);

	// intitialize position
	robots[rid].position.x += positions[rid].x;
	robots[rid].position.y += positions[rid].y;
    robots[rid].position.theta += positions[rid].theta;
	
	// initialize movement parameters 
	// turn
	robots[rid].movement.turn_left = - (MIN_TURN + (HRAND * (MAX_TURN - MIN_TURN)));
	robots[rid].movement.turn_right = (MIN_TURN + (HRAND * (MAX_TURN - MIN_TURN)));
	robots[rid].movement.turn_forward = (MIN_TURN + (HRAND * (MAX_TURN - MIN_TURN)));
	robots[rid].movement.turn_forward -= robots[rid].movement.turn_forward / 2;
	robots[rid].movement.turn_error = MAX_TURN_ERROR * HRAND;
	// speed
	robots[rid].movement.speed_left = (MIN_SPEED + (HRAND * (MAX_SPEED - MIN_SPEED)));
	robots[rid].movement.speed_right = (MIN_SPEED + (HRAND * (MAX_SPEED - MIN_SPEED)));
	robots[rid].movement.speed_forward = (MIN_SPEED + (HRAND * (MAX_SPEED - MIN_SPEED)));
	robots[rid].movement.speed_error = MAX_SPEED_ERROR * HRAND;
	
	// initialize comm parameters
	robots[rid].comm.comm_error = MAX_COMM_ERROR * HRAND; //probability of message not being transmitted 
	robots[rid].comm.range = (MIN_COMM_RANGE + (HRAND * (MAX_COMM_RANGE - MIN_COMM_RANGE)));
	robots[rid].comm.range_error = MAX_COMM_RANGE_ERROR * HRAND;
	
	//comms
	robots[rid].tx_flag=false;
	robots[rid].rx_flag=false;

	//motors
	robots[rid].left_motor=0;
	robots[rid].left_motor_active=false;
	robots[rid].right_motor=0;
	robots[rid].right_motor_active=false;
}