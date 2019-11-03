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

const int MAGNET_PIN_NUMBER = A3;
electromagnetCtrl eMagnet(MAGNET_PIN_NUMBER);

const int L_PRESS_SENSOR_PIN_NUMBER = 0;
pressSensor leftPress(L_PRESS_SENSOR_PIN_NUMBER);

const int R_PRESS_SENSOR_PIN_NUMBER = 1;
pressSensor rightPress(R_PRESS_SENSOR_PIN_NUMBER);


unsigned char z;
unsigned char catchOK;
int l;
int theta;

union readDataUnion{
  int integer_point;
  byte byte_point[2];
};
union readDataUnion x,y;

unsigned char nowMoving;

int thetaPercent;
int nowTurn;
int nowExtend;
unsigned char canCatch;

unsigned long int movingTime;
unsigned long int recordTime;
int turnRand;

void setup() {
  Serial.begin(115200);
  armExtend.initialize();
  armTurn.initialize();
  fan.initialize();
  eMagnet.initialize();
  x.integer_point=80;
  y.integer_point=100;
  z=60;
  l = calcLength(x.integer_point,y.integer_point);
  theta = calcDegree(x.integer_point,y.integer_point);
  randomSeed(analogRead(2));
  nowMoving=0;
  recordTime=0;
  movingTime=0;
  catchOK=0;
  canCatch=0;
  fan.on();
}

void loop() {
  /*動作テスト用*/
//  nowExtend =   armExtend.extend(110,90);//110mmで30mmの高さのところへ伸ばす
//  nowTurn   =   armTurn.turn(45);//20度回転
//  fan.on();//ファンの電源をOnに
//  eMagnet.on(); //マグネットをoffに
//  led.bright(0x1);
//eMagnet.on(); 
  /*本番用*/
  //情報更新
  if(Serial.available()>=5){
    /*'s'+x座標+y座標*/
    char readData=Serial.read();
    if(readData=='s'){                //ビッグエンディアンかリトルエンディアンかで[0]と[1]の場所を変更
      x.byte_point[1]=Serial.read();
      x.byte_point[0]=Serial.read();
      y.byte_point[1]=Serial.read();
      y.byte_point[0]=Serial.read();
      l     = calcLength(x.integer_point,y.integer_point);
      theta = calcDegree(x.integer_point,y.integer_point);
      z     = 100;//上が正
      nowMoving=0x01;
    }
  }
  nowTurn   =   armTurn.turn(theta);    //回る
  nowExtend =   armExtend.extend(l,z);  //伸ばす
  fanSwitch(movingTime,5,10);         //ファンを5*10秒間onにし，のちに5秒offにする
  switch(nowMoving){
    case 0x01:
      z=100;
      nowMoving=0x02;
    break;
    case 0x02:
      recordTime =millis();
      nowMoving =0x03;
    break;
    case 0x03:
      if((millis()-recordTime)>1000){
        nowMoving =0x11;
      }
    break;
    
    /*
     * キャッチ下げ
     */
    case 0x11: //電磁石オン
      eMagnet.on();
      nowMoving =0x12;
    break;
    case 0x12: //キャッチ下げ
      z-=1;
      nowMoving =0x13;
    break;
    case 0x13:
      if(z<=40){
        nowMoving = 0x14;
      }else{
        nowMoving=0x12;
        delay(2);  
      }
    break;
    case 0x14:
      recordTime =millis();
      nowMoving =0x15;
    break;
    case 0x15:
      if((millis()-recordTime)>1000){
        nowMoving =0x16;
      }
    break;
    case 0x16:
      if(nowTurn==0&&nowExtend==0){
        nowMoving =0x21;
      }
    break;
    /*
     * キャッチ上げ
     */
    case 0x21://キャッチ上げ
      z=80;
      nowMoving =0x22;
    break;
    case 0x22:
      recordTime =millis();
      nowMoving =0x23;
    break;
    case 0x23:
      if((millis()-recordTime)>1000){
        nowMoving =0x24;
      }
    break;
    case 0x24://回転のする角度や長さの決定
      turnRand = random(50,70);
      l = random(100,180);
      nowMoving = 0x25;
    break;
    case 0x25:
      if(nowTurn==0&&nowExtend==0){
        nowMoving =0x26;
      }
    break;
    case 0x26:
      if(nowExtend==0){
        nowMoving = 0x27;
      }
    break;
    case 0x27:
      recordTime =millis();
      nowMoving =0x28;
    break;
    case 0x28:
      if((millis()-recordTime)>1000){
        nowMoving =0x29;
      }
    break;
    case 0x29://回転方向の決定
      if(theta<90){
        nowMoving =0x31;//右→左
      }else{
        nowMoving =0x41;//左→右
      }
    break;
    /*
     * 右→左
     */
    case 0x31://回転
      theta = 180-turnRand;
      nowMoving =0x32;
    break;
    case 0x32:
      recordTime =millis();
      nowMoving =0x33;
    break;
    case 0x33:
      if((millis()-recordTime)>3000){
        nowMoving =0x34;
      }
    break;
    case 0x34:
      if(nowTurn==0&&nowExtend==0){
        nowMoving =0x51;
      }
    break;
    /*
     * 左→右
     */
    case 0x41://回転
      theta = turnRand;
      nowMoving =0x42;
    break;
    case 0x42:
      recordTime =millis();
      nowMoving =0x43;
    break;
    case 0x43:
      if((millis()-recordTime)>3000){
        nowMoving =0x44;
      }
    break;
    case 0x44:
      if(nowTurn==0&&nowExtend==0){
        nowMoving =0x61;
      }
    break;
    /*
      * 左で離す
     */
    case 0x51://初期化
      canCatch=0;
      recordTime =millis();
      nowMoving =0x52;
    break;
    
    case 0x52:
      eMagnet.off();
      while((millis()-recordTime)<1000){
        canCatch+=leftPress.reading(10);
        delay(2);
      }
      nowMoving =0x71;
    break;
    /*
     * 右で離す
     */
    case 0x61://初期化
      canCatch=0;
      recordTime =millis();
      nowMoving =0x62;
    break;
    
    case 0x62:
      eMagnet.off();
      while((millis()-recordTime)<1000){
        canCatch+=rightPress.reading(10);
        delay(2);
      }
      nowMoving =0x71;
    break;
    /*
     *  結果送信
     */
    case 0x71:
      if(canCatch>0){
        Serial.write(byte(11));
        Serial.flush();
      }else{
        Serial.write(byte(10));
        Serial.flush();
      }
      nowMoving =0x00;
      canCatch  =0;
    break;

    default:
    break;
  }
  movingTime = millis()/1000;
  delay(10);//動作安定のため
}

void fanSwitch(unsigned long int timer,unsigned long int clk_seconds,unsigned int rest_clk){
  if(timer%(clk_seconds*rest_clk)<clk_seconds){
    fan.on();
  }else{
    fan.off();
  }
}

int calcDegree(int x, int y){
  return((int)(atan2(y,x)*180.0/PI));
}
int calcLength(int x, int y){
  return((int)((double)sqrt((double)pow(x,2)+(double)pow(y,2))));
}
