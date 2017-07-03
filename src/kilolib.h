#pragma once
#ifndef KILOLIB_H
#define KILOLIB_H

#include "robot.h"

typedef void(*vFunctionCall)();
typedef double distance_measurement_t;

//communication data struct without distance it should be 9 bytes max
struct message_t {
    unsigned char type = 0;
    unsigned char data[9];
    unsigned char crc;
};

class Kilobot
{
	Robot *this_robot;
private:

public:
	uint32_t kilo_ticks = 0;
	unsigned int kilo_turn_left = 60;
	unsigned int kilo_turn_right = 60;
    unsigned int kilo_straight_left = 60;
    unsigned int kilo_straight_right = 60;

	// Implemented in kilobot.cpp
	virtual void setup() = 0;
	virtual void loop() = 0;
	virtual int main() = 0;
	virtual void message_rx(message_t *message, distance_measurement_t *distance_measurement) = 0;
	virtual void message_tx_success() = 0;
	virtual message_t *message_tx() = 0;

	void kilo_init() {};

	void init() {
        // TODO: I don't think this is ever used/called
		float two_hours = TICS_PER_SECOND * 60 * 60 * 2;
		//this_robot.battery = (1 + gauss_rand(rand())/5) * two_hours;
        this_robot->battery = two_hours;
		setup();
	}

	unsigned int rand_soft() {
		return rand() * 255 / RAND_MAX;
	}

	uint8_t rand_hard() {
		return (uint8_t) rand() % 255;
	}

	unsigned char message_crc(message_t *m) {
		int crc = 0;
		for (int i=0; i<9; i++) {
			crc += m->data[i];
		} return crc % 256;
	}

	void set_color(Led_Color c) {
		this_robot->led = c;
	};

	void spinup_motors() {
		this_robot->left_motor_active = true;
		this_robot->right_motor_active = true;
	};

	void set_motors(unsigned int l, unsigned int r)
	{
		this_robot->left_motor = l;
		this_robot->right_motor = r;
	};

	Led_Color RGB(float r, float g, float b) {
        Led_Color tmp;
        tmp.R = r;
        tmp.G = g;
        tmp.B = b;
		return tmp;
	};

	void run_controller(Robot *r) {
		this_robot = r;

		if (this_robot->message_sent) {
			this_robot->tx_flag = false;
			this_robot->message_sent = false;
			message_tx_success();
		}
		// Implement time step with noise
		kilo_ticks++;
		int rand_tick = rand();
		if (rand_tick < RAND_MAX * .1) {
			if (rand_tick < RAND_MAX * .05) {
				kilo_ticks--;
			} else {
				kilo_ticks++;
			}
		}

		// Run kilobot loop
		loop();

		// Implement effects of kilobot loop
		this_robot->motor_command = 4;
		// TODO: Does this follow *actual* left/right motor or turning convention?
		if (this_robot->right_motor_active && this_robot->right_motor == kilo_turn_right) {
			this_robot->motor_command -= 2;
		} else {
			this_robot->right_motor_active = false;
		}
		if (this_robot->left_motor_active && this_robot->left_motor == kilo_turn_left) {
			this_robot->motor_command  -= 1;
		} else {
			this_robot->left_motor_active = false;
		}
		if (message_tx()) {
			this_robot->tx_flag = true;
		} else {
			this_robot->tx_flag = false;
		}
	};

	unsigned int estimate_distance(distance_measurement_t *d) {
		if (*d < 255) {
			return (unsigned int) *d;
		} else {
			return 255;
		}
	}

	void received() {
		this_robot->message_sent = true;
	}
};

#endif
