#include "pressSensor.h"

pressSensor::pressSensor(unsigned int pinNum){
  pin=pinNum;
}

int pressSensor::reading(unsigned int thVoltage){
  if(analogRead(pin)>=thVoltage){
    return 1;
  }else{
    return 0;
  }
}
