#ifndef _PINS_STRUCTURES_H_
  #define _PINS_STRUCTURES_H_
  struct extendParameters{
    int pinNum;
    int maxDownDegree;
    int maxUpDegree;
  };

  struct turnParameters{
    int pinNum;
    int maxRightDegree;
    int maxLeftDegree;
  };
#endif
