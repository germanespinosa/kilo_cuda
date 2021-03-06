#include "kilolib.h"
class Kilo_Impl : public Kilobot {
public:

void setup() {
	// put your setup code here, to be run only once
	spinup_motors();
}

void loop() {
	// put your main code here, to be run repeatedly
	static int step = 0;

	if (!(kilo_ticks ))
	{
		step++;
		if (step )
		{
			set_motors(0, 0);
			set_color(RGB(1, 0, 0));
		}
		else
		{
			spinup_motors();
			set_motors(kilo_turn_left, kilo_turn_right);
			set_color(RGB(0, 1, 0));
		}
	}
}

int main() {
	// initialize hardware
	kilo_init();
	// start program
	//kilo_start(setup, loop);

	return 0;
}};
