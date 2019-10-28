#include "fanCtrl.h"
fanCtrl::fanCtrl(unsigned int pinNum){
  pin = pinNum;
}
void fanCtrl::initialize(void){
  pinMode(pin,OUTPUT);  
}
void fanCtrl::on(void){
  digitalWrite(pin,LOW);
}
void fanCtrl::off(void){
  digitalWrite(pin,HIGH);
}
