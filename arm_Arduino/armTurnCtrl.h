#ifndef _ARM_TURN_CTRL_H_
  #define _ARM_TURN_CTRL_H_
  #include "arduino.h"
  #include "pinsStructures.h"
  #include <Servo.h>
  class armTurnCtrl{
    public:
      armTurnCtrl(struct turnParameters turnParameters);
      ~armTurnCtrl(void);
      void initialize(void);
      int turn(int deg);
    private:
      int renewDeg(int deg);
      struct turnParameters turnPara;
      Servo *turnServo;
  };
#endif
