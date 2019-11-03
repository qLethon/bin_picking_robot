#include "electromagnetCtrl.h"
electromagnetCtrl::electromagnetCtrl(unsigned int pinNum){
  pin = pinNum;
}
void electromagnetCtrl::initialize(void){
  pinMode(pin,OUTPUT);  
}
void electromagnetCtrl::on(void){
  digitalWrite(pin,HIGH);
}
void electromagnetCtrl::off(void){
  digitalWrite(pin,LOW);
}
