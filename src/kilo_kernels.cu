#include "kilo_kernels.h"
#include "util.h"
#include "kilolib.h"
#include "kilocode.cpp"
#include <curand.h>
#include <curand_kernel.h>
#include <pthread.h>

static Robot         *cuda_robots;
static Position      *cuda_next_positions;
static Rectangle     *cuda_light_shapes;
static curandState_t *cuda_rand_states;

static Robot         *local_robots;
static Position      *local_positions;

//kernels prototypes

__global__ void compute_step_kernel(Robot *robots, Position *next_position, curandState_t *rand_states);
__global__ void collision_and_comms_kernel(Robot *robots,Position *next_position, curandState_t *rand_states);
__global__ void update_state_kernel(Robot *robots,Position *next_positio, curandState_t *rand_statesn);
__global__ void initialize_robot_data_kernel(Robot *robots, Position *positions, curandState_t *rand_states);
__global__ void compute_step_kernel(Robot *robots, Position *next_position, curandState_t *rand_states);
__global__ void compute_light_kernel(Robot *robots, Rectangle *cuda_light_shapes, curandState_t *rand_states);

static dim3 lingrid(LINGRID,1);
static dim3 block(TILELIMIT,1);
static dim3 shapesgrid;
pthread_t controller_thread[THREADS];
int threadids[THREADS];
pthread_mutex_t controller_mutexes[THREADS];
bool running = false;

void *execute_controllers(void *tid_ptr)
{
	Kilo_Impl kilobot;
	int tid = *((int *)tid_ptr);
	printf("%d started\n", tid);
	int rb = ROBOTS / THREADS * tid;
	int rf = tid == THREADS -1? ROBOTS : ROBOTS / THREADS * (tid + 1);
	printf("range %d %d\n", rb,rf);
	while (running)
	{
		pthread_mutex_lock(controller_mutexes + tid);
		//printf("%d waking up\n", tid);
		for (int rid = rb; rid < rf; rid++)
		{
			kilobot.run_controller(local_robots + rid);
		}
		pthread_mutex_unlock(controller_mutexes + tid);
	}
	return NULL;
}

void initialize_shapes(Rectangle *rectangles, int shapecount)// add upload shapes.
{
    cudaMalloc((void**)&cuda_light_shapes, sizeof(Rectangle) * shapecount);
	dim3 grid(LINGRID,shapecount);
	shapesgrid = grid;
 	cudaMemcpy(cuda_light_shapes, rectangles, sizeof(Rectangle) * shapecount, cudaMemcpyHostToDevice);
}
// floats(x1,y1)(x2,y2) int(r,g,b)

void initialize_positions(Position *positions)
{
	local_positions = (Position *)malloc(sizeof(Position)*ROBOTS);
	HANDLE_ERROR(cudaMalloc((void**)&cuda_next_positions, sizeof(Position) * ROBOTS));
	HANDLE_ERROR(cudaMemcpy(cuda_next_positions, positions, sizeof(Position) * ROBOTS, cudaMemcpyHostToDevice));
}
void initialize_robots(Position *positions) //
{
	local_robots = (Robot *)malloc(sizeof(Robot)*ROBOTS);
	initialize_positions(positions);
	HANDLE_ERROR(cudaMalloc((void**)&cuda_robots, sizeof(Robot) * ROBOTS));
	HANDLE_ERROR(cudaMalloc((void**)&cuda_rand_states, sizeof(curandState_t) * ROBOTS * 2));
	initialize_robot_data_kernel <<< lingrid, block >>> (cuda_robots, cuda_next_positions, cuda_rand_states);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	running = true;
	for (int tid = 0; tid < THREADS; tid++)
	{
		controller_mutexes[tid] = PTHREAD_MUTEX_INITIALIZER;
		threadids[tid] = tid;
		if (pthread_create(controller_thread+tid, NULL, execute_controllers, threadids+tid)) {
			fprintf(stderr, "Error creating thread\n");
		}
	}
}

void simulation_step()
{
	//Kilo_Impl kilobot;
	compute_step_kernel <<< lingrid, block >>> ( cuda_robots, cuda_next_positions, cuda_rand_states);
	//compute_light <<< shapesgrid, block >>> ( cuda_robots, cuda_light_shapes );
	collision_and_comms_kernel <<< lingrid, block >>> ( cuda_robots, cuda_next_positions, cuda_rand_states);
	//collision_and_comms << < lingrid, block >> > (cuda_robots, cuda_next_positions);
	update_state_kernel <<< lingrid, block >>> ( cuda_robots, cuda_next_positions, cuda_rand_states);
	// download data
	for (int tid = 0; tid < THREADS; tid++)
		pthread_mutex_lock(controller_mutexes + tid);
	HANDLE_ERROR(cudaMemcpy(local_robots, cuda_robots, sizeof(Robot) * ROBOTS, cudaMemcpyDeviceToHost));
	for (int tid = 0; tid < THREADS; tid++)
		pthread_mutex_unlock(controller_mutexes + tid);

	for (int tid = 0; tid < THREADS; tid++)
		pthread_mutex_lock(controller_mutexes + tid);
	HANDLE_ERROR(cudaMemcpy(cuda_robots, local_robots, sizeof(Robot) * ROBOTS, cudaMemcpyHostToDevice));
	// upload state changes
	for (int tid = 0; tid < THREADS; tid++)
		pthread_mutex_unlock(controller_mutexes + tid);
}

Position *download_position_data()
{
	cudaMemcpy(local_positions, cuda_next_positions, sizeof(Position) * ROBOTS, cudaMemcpyDeviceToHost);
	return local_positions;
}


Robot *download_robot_data()
{
 	cudaMemcpy(local_robots, cuda_robots, sizeof(Robot) * ROBOTS, cudaMemcpyDeviceToHost);
    return local_robots;
}

void release_cuda_memory()
{
	running = false;
	cudaFree(cuda_robots);
	cudaFree(cuda_next_positions);
	cudaFree(cuda_robots);
	cudaFree(cuda_rand_states);
	free(local_positions);
	free(local_robots);
	cudaThreadExit();
}

__global__ void compute_light_kernel(Robot *robots, Rectangle *light_shapes, curandState_t *rand_states)
{
//    unsigned int sid = SHAPEID;
//	unsigned int rid = ROBOTID;
	//if robot rid is in shape sid 
	// robot[rid] light = value. 
}

__global__ void compute_step_kernel(Robot *robots, Position *next_position, curandState_t *rand_states)
{
    unsigned int rid = ROBOTID;
	//compute the movement needed	
	Step step;
	if (robots[rid].left_motor == 0) robots[rid].left_motor_active = false;
	if (robots[rid].right_motor == 0) robots[rid].right_motor_active = false;

	float turn_error  = HRAND * robots[rid].movement.turn_error - robots[rid].movement.turn_error / 2;
	float speed_error = HRAND * robots[rid].movement.speed_error - robots[rid].movement.speed_error / 2;
	
	step.turn = robots[rid].left_motor_active ? (robots[rid].right_motor_active ? robots[rid].movement.turn_forward: robots[rid].movement.turn_left) : (robots[rid].right_motor_active ? robots[rid].movement.turn_right: 0);
	step.speed = robots[rid].left_motor_active ? (robots[rid].right_motor_active ? robots[rid].movement.speed_forward: robots[rid].movement.speed_left) : (robots[rid].right_motor_active ? robots[rid].movement.speed_right: 0);;
	
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

__global__ void collision_and_comms_kernel(Robot *robots,Position *next_position, curandState_t *rand_states)
{
	unsigned int rid = ROBOTID;
//	unsigned int nid = blockIdx.x;
	bool colide = false;
	for (int nid = 0; nid < rid && !colide; nid++)
	{
		float d = DIST(next_position[rid],robots[nid].position);
		if (d < ROBOT_RADIUS)
		{
			next_position[rid] = robots[rid].position;
			colide = true;
		}
	}
	if (robots[rid].tx_flag)
	{
		float range_error = robots[rid].comm.range_error*HRAND - robots[rid].comm.range_error / 2;
		float range = robots[rid].comm.range + range_error;
		for (int nid = 0; nid < ROBOTS ; nid++)
		{
			float d = DIST(next_position[rid], robots[nid].position);
			if (d < range)
			{
				if (HRAND > robots[rid].comm.comm_error)
				{
					robots[nid].rx_flag = true;
					robots[nid].rx_distance = d + range_error;
					for (int i = 0; i < MESSAGE_SIZE; i++)
					{
						robots[nid].message_rx[i] = robots[rid].message_rx[i];
					}
				}
			}
		}
	}
}

__global__ void update_state_kernel(Robot *robots,Position *next_position, curandState_t *rand_states)
{
    unsigned int rid = ROBOTID;
	robots[rid].position=next_position[rid];
	robots[rid].tx_flag=false;
}

__global__ void initialize_robot_data_kernel(Robot *robots, Position *positions, curandState_t *rand_states)
{
    unsigned int rid = ROBOTID;
	// initialize random seeds
	// all robots have the same soft random sequence
	curand_init(0, 0, 0, rand_states + rid);

	// all robots have different hard random sequence
	curand_init(rid, 0, 0, rand_states + ROBOTS + rid );

	// intitialize position
	robots[rid].position.theta = positions[rid].theta;
	robots[rid].position.x = positions[rid].x;
	robots[rid].position.y = positions[rid].y;
	
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