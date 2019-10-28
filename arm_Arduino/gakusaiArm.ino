#include "armExtendCtrl.h"
#include "pinsStructures.h"
#include "fanCtrl.h"
#include "electromagnetCtrl.h"
#include "armTurnCtrl.h"
#include "pressSensor.h"

struct extendParameters handParametaersStruct     ={8,0,180};
struct extendParameters elbowPatametersStruct     ={10,180,0};
struct extendParameters shoulderPatametersStruct  ={9,0,90};
armExtendCtrl armExtend(handParametaersStruct,elbowPatametersStruct,shoulderPatametersStruct);

struct turnParameters turnParametaersStruct       ={11,0,180};
armTurnCtrl   armTurn(turnParametaersStruct);

const int FAN_PIN_NUMBER = 12;
fanCtrl fan(FAN_PIN_NUMBER);

const int MAGNET_PIN_NUMBER = 13;
electromagnetCtrl eMagnmet(MAGNET_PIN_NUMBER);

const int L_PRESS_SENSOR_PIN_NUMBER = 1;
pressSensor leftPress(L_PRESS_SENSOR_PIN_NUMBER);

const int R_PRESS_SENSOR_PIN_NUMBER = 2;
pressSensor rightPress(R_PRESS_SENSOR_PIN_NUMBER);

unsigned char x,y,z;
unsigned char catchOK;
int l;
int theta;

unsigned char nowMoving;

int thetaPercent;
int nowTurn;
int nowExtend;
int canCatch;

unsigned long int movingTime;
unsigned long int recordTime;
int turnRand;

void setup() {
  armExtend.initialize();
  armTurn.initialize();
  fan.initialize();
  eMagnmet.initialize();
  x=100;
  y=100;
  z=0;
  l = calcLength(x,y);
  l = calcLength(x,y);
  theta = calcDegree(x,y);
  Serial.begin(115200);
  randomSeed(analogRead(2));
  nowMoving=0;
  recordTime=0;
  movingTime=0;
  catchOK=0;
  canCatch=0;
}

void loop() {
  /*動作テスト用*/
//  nowExtend =   armExtend.extend(70,30);//110mmで30mmの高さのところへ伸ばす
//  nowTurn   =   armTurn.turn(45);//20度回転
//  fan.on();//ファンの電源をOnに
//  eMagnmet.on(); //マグネットをoffに
  /*本番用*/
  
  if(Serial.available()>=4){
    /*'s'+x座標+y座標+'e'*/
    unsigned char readData=Serial.read();
    if(readData=='s'){
      x=Serial.read();
      y=Serial.read();
    }
    l = calcLength(x,y);
    theta = calcDegree(x,y);
    z = 80;
    nowMoving=0x01;
  }
  nowTurn   =   armTurn.turn(theta);
  nowExtend =   armExtend.extend(l,z);
  fanSwitch(movingTime,5,10);
  switch(nowMoving){
    /*
     * キャッチ作業
     */
    case 0x01: //電磁石オン
      eMagnmet.on();
      nowMoving =0x02;
    break;
    case 0x02: //キャッチ下げ
      if(nowTurn==0&&nowExtend==0){
        z--;
      }
      if(z==10){
        nowMoving =0x03;  
      }
    break;
    case 0x03://キャッチ上げ
      if(nowTurn==0&&nowExtend==0){
        z++;
      }
      if(z==50){
        nowMoving =0x04;
      }
    break;
    case 0x04:
      turnRand = random(10,80);
      nowMoving = 0x55;
    break;
    case 0x05://回転方向の決定
      if(theta<90){
        nowMoving =0x11;//右→左
      }else{
        nowMoving =0x21;//左→右
      }
    break;
    /*
     * 右→左
     */
    case 0x11://回転
      if(nowTurn==0&&nowExtend==0){
        theta++;
      }
      if(theta>(90)+turnRand){
        nowMoving =0x31;
      }
    break;
    /*
     * 左→右
     */
    case 0x21://回転
      if(nowTurn==0&&nowExtend==0){
        theta--;
      }
      if(theta>turnRand){
        nowMoving =0x41;
      }
    break;
    /*
     * 左で離す
     */
    case 0x31://初期化
      canCatch=0;
      nowMoving =0x32;
    break;

    case 0x32:
      recordTime =millis();
      canCatch+=leftPress.reading(20);
      nowMoving =0x33;
    break;

    case 0x33:
      canCatch+=leftPress.reading(20);
      eMagnmet.off();
      if((millis()-recordTime)>1000){
        nowMoving =0x51;
      }
    break;
    /*
     * 右で離す
     */
    case 0x41://初期化
      canCatch=0;
      nowMoving =0x42;
    break;

    case 0x42:
      recordTime =millis();
      canCatch+=rightPress.reading(20);
      nowMoving =0x43;
    break;

    case 0x43:
      canCatch+=rightPress.reading(20);
      eMagnmet.off();
      if((millis()-recordTime)>1000){
        nowMoving =0x51;
      }
    break;

    /*
     *  結果送信
     */
    case 0x51:
      if(canCatch>0){
        Serial.write('o');
      }else{
        Serial.write('x');
      }
      nowMoving =0x00;
    break;
    default:
    break;
  }
  movingTime = millis()/1000;
  delay(1);//動作安定のため
}

void fanSwitch(unsigned long int timer,unsigned long int clk_seconds,unsigned int rest_clk){
  if(timer%(clk_seconds*rest_clk)<clk_seconds){
    fan.on();
  }else{
    fan.off();
  }
}

int calcDegree(unsigned char x, unsigned char y){
  return((int)(atan2(x,y)*180.0/PI));
}
int calcLength(unsigned char x, unsigned char y){
  return((int)(sqrt(pow(x,2)+pow(y,2))));
}
