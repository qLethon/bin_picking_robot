#ifndef _FAN_CTRL_H_
  #define _FAN_CTRL_H_
  #include "arduino.h"
  class fanCtrl{
    public:
      fanCtrl(unsigned int pinNum);
      void initialize(void);
      void on(void);
      void off(void);
    private:
      unsigned int pin;
  };
#endif
