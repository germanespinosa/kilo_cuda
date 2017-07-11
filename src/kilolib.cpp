#include "kilolib.h"

void Kilobot::kilo_init()
{
};
void Kilobot::run_controller(Robot *r)
{
	this_robot = r;
	loop();
};
void Kilobot::run_setup(Robot *r)
{
	this_robot = r;
	setup();
};
uint8_t	Kilobot::estimate_distance(const distance_measurement_t *d)
{
	return this_robot->rx_distance;
};
uint8_t Kilobot::rand_hard()
{
	static int shared_counter = 0;
	srand(shared_counter++);
	return rand() % 256;
};
uint8_t Kilobot::rand_soft()
{
	srand(this_robot->soft_rand_counter++);
	return rand() % 256;
};
void Kilobot::rand_seed(uint8_t seed)
{
	this_robot->soft_rand_counter = seed;
};
int16_t Kilobot::get_ambientlight()
{
	return this_robot->ambient_light;
};
int16_t Kilobot::get_voltage()
{
	return this_robot->battery;
};
int16_t Kilobot::get_temperature()
{
	return 1200;
};
void Kilobot::set_motors(uint8_t left, uint8_t right)
{
	this_robot->left_motor = left;
	this_robot->right_motor = right;
};
void Kilobot::spinup_motors()
{
	this_robot->left_motor_active = true;
	this_robot->right_motor_active = true;
};
void Kilobot::set_color(uint8_t color)
{
	this_robot->led = color;
};
uint16_t  Kilobot::message_crc(const message_t *msg)
{
	return CRC(msg->data);
};
