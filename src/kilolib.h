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
};

class Kilobot
{
	Robot *this_robot;
private:
	typedef void(*message_rx_t)(message_t *, distance_measurement_t *d);
	typedef message_t *(*message_tx_t)(void);
	typedef void(*message_tx_success_t)(void);
public:
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