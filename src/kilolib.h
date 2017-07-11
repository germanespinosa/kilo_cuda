<<<<<<< HEAD
#ifndef KILOLIB
#define KILOLIB
#include <stdlib.h>  
#include "robot.h"

struct distance_measurement_t
{
	int16_t low_gain, hi_gain;
};

struct message_t
{
	uint8_t data[9];
	uint8_t type;
	uint8_t crc;
=======
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
>>>>>>> 30e0ae1723acd6948008a6ce9871206ef2fd7eab
};

class Kilobot
{
	Robot *this_robot;
private:
	typedef void(*message_rx_t)(message_t *, distance_measurement_t *d);
	typedef message_t *(*message_tx_t)(void);
	typedef void(*message_tx_success_t)(void);
public:
<<<<<<< HEAD
	typedef void(*Loop)(void);
	volatile uint32_t kilo_ticks;
	volatile uint16_t kilo_tx_period;
	uint16_t kilo_uid;
	uint8_t kilo_turn_left;
	uint8_t kilo_turn_right;
	uint8_t kilo_straight_left;
	uint8_t kilo_straight_right;
	message_rx_t kilo_message_rx;
	message_tx_t kilo_message_tx;
	message_tx_success_t kilo_message_tx_success;
	virtual void setup() = 0;
	virtual void loop() = 0;
	virtual int main() = 0;
	void kilo_init();
	uint8_t	estimate_distance(const distance_measurement_t *d);
	uint8_t rand_hard();
	uint8_t rand_soft();
	void rand_seed(uint8_t seed);
	int16_t get_ambientlight();
	int16_t get_voltage();
	int16_t get_temperature();
	void set_motors(uint8_t left, uint8_t right);
	void spinup_motors();
	void set_color(uint8_t color);
	uint16_t  message_crc(const message_t *msg);
	void run_controller(Robot *r);
	void run_setup(Robot *r);
};
#endif
=======
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
>>>>>>> 30e0ae1723acd6948008a6ce9871206ef2fd7eab
