#ifndef _PRESS_SENSOR_H_
  #define _PRESS_SENSOR_H_
  #include "arduino.h"
  class pressSensor{
    public:
      pressSensor(unsigned int pinNum);
      int reading(unsigned int thVoltage);
    private:
      unsigned int pin;
  };
#endif
