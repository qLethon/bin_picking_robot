#include "pressSensor.h"

pressSensor::pressSensor(unsigned int pinNum){
  pin=pinNum;
}

int pressSensor::reading(unsigned int thVoltage){
  if(analogRead(pin)>=(int)thVoltage){
    return 1;
  }else{
    return 0;
  }
}
