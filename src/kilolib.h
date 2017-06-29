#include "robot.h"
typedef void(*vFunctionCall)();

class Kilobot
{
	Robot *this_robot;
private:

public:
	long kilo_ticks = 0;
	unsigned int kilo_turn_left = 255;
	unsigned int kilo_turn_right = 255;
	virtual void setup() = 0;
	virtual void loop() = 0;
	virtual int main() = 0;

	void kilo_init() {};

	void spinup_motors()
	{
		this_robot->left_motor_active = true;
		this_robot->right_motor_active = true;
	};

	void set_motors(unsigned int l, unsigned int r)
	{
		this_robot->left_motor = l;
		this_robot->right_motor = r;
	};

	void set_color(Led_Color c) {
		this_robot->led = c;
	};

	Led_Color RGB(float r, float g, float b) {
        Led_Color tmp;
        tmp.R = r;
        tmp.G = g;
        tmp.B = b;
		return tmp;
	};

	void run_controller(Robot *r)
	{
		this_robot = r;
		loop();
	};

};