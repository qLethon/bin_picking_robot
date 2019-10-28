#ifndef _ELECTROMAGNET_CTRL_H_
  #define _ELECTROMAGNET_CTRL_H_
  #include "arduino.h"
  class electromagnetCtrl{
    public:
      electromagnetCtrl(unsigned int pinNum);
      void initialize(void);
      void on(void);
      void off(void);
    private:
      unsigned int pin;
  };
#endif
