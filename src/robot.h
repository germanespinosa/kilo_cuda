#ifndef ROBOT
#define ROBOT

#define uint8_t unsigned char
#define int16_t short int
#define uint16_t unsigned short int
#define uint32_t unsigned int

#define RGB(R__, G__, B__) (R__ & 3 << 4) + (G__ & 3 << 2) + (B__ & 3)
#define CRC(D__) (D__[0] & 3 << 14) + (D__[1] & 3 << 12) + (D__[2] & 3 << 10) + (D__[3] & 3 << 8) + (D__[4] & 3 << 6) + (D__[5] & 3 << 4) + (D__[6] & 3 << 2) + (D__[7] & 3)

#define ROBOTS 512
#define ARENA_WIDTH 3000
#define ARENA_HEIGHT 3000
#define ROBOT_RADIUS 32
#define TICS_PER_SECOND 32
#define STEPS TICS_PER_SECOND * SECONDS

#define PI 3.1415
#define MIN_TURN .05
#define MAX_TURN .1
#define MAX_TURN_ERROR .01

#define MAX_SPEED .7
#define MIN_SPEED .3
#define MAX_SPEED_ERROR .01

#define MAX_COMM_ERROR .01
#define MIN_COMM_RANGE 45
#define MAX_COMM_RANGE 65
#define MAX_COMM_RANGE_ERROR 10

#define MESSAGE_SIZE 9

struct Movement_Parameters
{
	float turn_left;
	float turn_right;
	float turn_forward;
	float turn_error;
	float speed_left;
	float speed_right;
	float speed_forward;
	float speed_error;
};

struct Comm_Parameters
{	
	float comm_error;
	float range;	
	float range_error;
};

struct Position
{
    float x;     //between RADIUS and ARENA_WIDTH-RADIUS
    float y;     //between RADIUS and ARENA_HEIGHT_RADIUS
    float theta; //between 0 and 2 PI
};

struct Step
{
    float turn;
    float speed;
};

struct Robot
{
	Position position;
	Movement_Parameters movement;
	Comm_Parameters comm;
	unsigned char message_tx[MESSAGE_SIZE];
	unsigned char message_rx[MESSAGE_SIZE];
	bool tx_flag, rx_flag;
	uint8_t left_motor, right_motor;
	uint8_t rx_distance;
	bool left_motor_active, right_motor_active;
	uint8_t led;
	int soft_rand_counter;
	int16_t ambient_light;
	int16_t battery;
};

struct Point {
    float x;
    float y;
};

struct Rectangle {
    Point pos;  // lower right corner
    float width;
    float height;
    short color[3];
};;

#endif