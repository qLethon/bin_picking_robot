#ifndef _ARM_EXTEND_CTRL_H_
  #define _ARM_EXTEND_CTRL_H_
  #include "arduino.h"
  #include "pinsStructures.h"
  #include <Servo.h>
  class armExtendCtrl{
    public:
      armExtendCtrl(struct extendParameters handParameters,struct extendParameters elbowParameters,struct extendParameters shoulderParameters);
      ~armExtendCtrl(void);
      void initialize(void);
      int extend(int l,int depth);
    private:
      int overTurnRange(struct extendParameters para, int targetDegree);
      int nextDegree(Servo* ser,struct extendParameters para,int& nowDegree, int targetDegree);
      struct extendParameters handPara;
      struct extendParameters elbowPara;
      struct extendParameters shoulderPara;
      Servo *handServo;
      Servo *elbowServo;
      Servo *shoulderServo;
  };
#endif
