#include "armExtendCtrl.h"

static int nowHandDegree;
static int nowElbowDegree;
static int nowShoulderDegree;

struct armLengthStruct{
  const int SHOULDER2ELBOW  = 110;
  const int ELBOW2HAND      = 100;
}armLength;

armExtendCtrl::armExtendCtrl(struct extendParameters handParameters,struct extendParameters elbowParameters,struct extendParameters shoulderParameters){
  handPara      = handParameters;
  elbowPara     = elbowParameters;
  shoulderPara  = shoulderParameters;

  nowHandDegree     = 0;
  nowElbowDegree    = 90;
  nowShoulderDegree = 90;
  
  handServo     = new Servo;
  elbowServo    = new Servo;
  shoulderServo = new Servo;
}
armExtendCtrl::~armExtendCtrl(void){
  delete  handServo;
  delete  elbowServo;
  delete  shoulderServo;
}

void armExtendCtrl::initialize(void){
  handServo->attach(handPara.pinNum);
  elbowServo->attach(elbowPara.pinNum);
  shoulderServo->attach(shoulderPara.pinNum);
}

int armExtendCtrl::overTurnRange(struct extendParameters para, int targetDegree){
  int maxDeg  = (para.maxDownDegree<para.maxUpDegree)?(para.maxUpDegree):(para.maxDownDegree);
  int minDeg  = (para.maxDownDegree>para.maxUpDegree)?(para.maxUpDegree):(para.maxDownDegree);
  if(targetDegree>maxDeg){
    return 1;
  }else if(targetDegree<minDeg){
    return -1;
  }else{
    return 0;
  }
}
int armExtendCtrl::nextDegree(Servo* ser,struct extendParameters para,int& nowDegree, int targetDegree){
  switch(overTurnRange(para,targetDegree)){
    case 1: //範囲より大きい角度の場合
    case -1: //範囲より小さい角度の場合
      return -1;
    break;
    default:
      if(targetDegree>nowDegree){
        ser->write(++nowDegree);
        return 1;
      }else if(targetDegree<nowDegree){
        ser->write(--nowDegree);
        return 1;
      }else{
        ser->write(nowDegree);
        return 0;
      }
    break;
  }
}
int armExtendCtrl::extend(int l,int depth){
  int maxLength = armLength.SHOULDER2ELBOW+armLength.ELBOW2HAND;
  int wantLength = (int)(sqrt(pow(l,2)+pow(depth,2)));
  int targetHandDeg =0;
  int targetElbowDeg =0;
  int targetShoulderDeg =0;
  //深さの傾き分を出す
  int depthDeg        =  (int)((atan2(depth,l)*180.0)/PI);
  if(wantLength>maxLength){
    targetHandDeg = 0;
    targetElbowDeg =0;
    targetShoulderDeg =depthDeg;
  }else{
    //ここで余弦なんたら？で角度だす
    int elbowAbsoluteDeg     = (int)((acos(((double)(pow(armLength.SHOULDER2ELBOW,2)+pow(armLength.ELBOW2HAND,2)-pow(wantLength,2)))\
                                  /((double)(2.0*armLength.SHOULDER2ELBOW*armLength.ELBOW2HAND))))*180.0/PI);
    int shoulderAbsoluteDeg  = (int)((acos(((double)(pow(armLength.SHOULDER2ELBOW,2)+pow(wantLength,2)-pow(armLength.ELBOW2HAND,2)))\
                                  /((double)(2.0*armLength.SHOULDER2ELBOW*wantLength))))*180.0/PI);
    if(shoulderAbsoluteDeg>90)shoulderAbsoluteDeg=180-shoulderAbsoluteDeg;
    int handAbsoluteDeg   = 180-elbowAbsoluteDeg-shoulderAbsoluteDeg;
    //手の角度変換
    targetHandDeg = handAbsoluteDeg;
    targetHandDeg -= depthDeg;
    targetElbowDeg = 180-elbowAbsoluteDeg;
    targetShoulderDeg = shoulderAbsoluteDeg;
    targetShoulderDeg += depthDeg;
    if(targetHandDeg<0)targetHandDeg=0;
    if(targetElbowDeg<0)targetElbowDeg=0;
    if(targetShoulderDeg<0)targetShoulderDeg=0;
  }
  //回転
  int rt =0;
  switch(nextDegree(handServo,handPara,nowHandDegree,targetHandDeg)){
    case -1:
      return -1;
    break;
    case 1:
      rt=1;
    break;
    default:
    break;
  }
  switch(nextDegree(elbowServo,elbowPara,nowElbowDegree,targetElbowDeg)){
    case -1:
      return -1;
    break;
    case 1:
      rt=1;
    break;
    default:
    break;
  }
  switch(nextDegree(shoulderServo,shoulderPara,nowShoulderDegree,targetShoulderDeg)){
    case -1:
      return -1;
    break;
    case 1:
      rt=1;
    break;
    default:
    break;
  }
  return rt;
}
