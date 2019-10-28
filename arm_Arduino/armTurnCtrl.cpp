#include "armTurnCtrl.h"
static int nowTurnDegree;
armTurnCtrl::armTurnCtrl(struct turnParameters turnParameters){
  turnPara =   turnParameters;
  nowTurnDegree=90;
  turnServo = new Servo;
}

armTurnCtrl::~armTurnCtrl(void){
  delete turnServo;
}

void armTurnCtrl::initialize(void){
  turnServo->attach(turnPara.pinNum);
}

int armTurnCtrl::turn(int deg){
  deg+=5;
  int rt=0;
  if(turnPara.maxRightDegree<turnPara.maxLeftDegree){
    rt=renewDeg(deg);
  }else{
    rt=renewDeg(180-deg);
  }
  turnServo->write(nowTurnDegree);
  return rt;
}

int armTurnCtrl::renewDeg(int targetDeg){
  if(targetDeg<nowTurnDegree){
    nowTurnDegree--;
    return 1;
  }else if(targetDeg>nowTurnDegree){
    nowTurnDegree++;
    return 1;
  }else{
    return 0;  
  }
}
