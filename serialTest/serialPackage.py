#-*- coding: utf-8 -*-
import serial
import time
class armCommunication:
	def __init__(self,portName,rate,timeoutSeconds):
		self.ser = serial.Serial(portName, rate ,timeout = timeoutSeconds,parity=serial.PARITY_NONE) 
		self.ser.close()
		time.sleep(1)
		self.ser.open() #開く(ポートを開閉→マイコン側の再起動にも繋がるので注意)
		time.sleep(3) #Arduino側の再起動待機のため4秒停止(これは2~3秒もしくはそれ以下でも良いかも？(試して決定))
		self.ser.reset_input_buffer()#受信バッファのクリア
	
	def send_position(self,x,y):
		byteSendData =['s'.encode('utf-8')] 
		byteSendData.append(((x&0xff00)>>8).to_bytes(1,'big'))
		byteSendData.append((x&0xff).to_bytes(1,'big'))
		byteSendData.append(((y&0xff00)>>8).to_bytes(1,'big'))
		byteSendData.append((y&0xff).to_bytes(1,'big'))
		for b in byteSendData:
			self.ser.write(b)#送信
			self.ser.flush()#送信完了まで待機
			time.sleep(0.1)#安定のための待機

	def read_one_byte(self):
		try :
			byteReveiveData = self.ser.read(1)#1byte受信(1行読み込むならreadline)
			reseiveData = int.from_bytes(byteReveiveData, 'big')#byte型に変換
			return reseiveData;
		except : #タイムアウト(USBが接続されてないと出すらしい)(「何かしらのデータを受信しなかったら」ではない)
			print("error") #ここで停止はしない
			return -1