#-*- coding: utf-8 -*-
import serial
import time

#Serial通信のデモプログラム
#スレッド立てるなら#★以前の処理は避けてくださいな(初期化みたいなモノなので)

def pick(x, y):
	# ポート番号・ポートレート・タイムアウト時間を設定しポートをオープン
	#また，パリティビットは使用しない
	ser = serial.Serial('COM3', 115200 ,timeout = 30,parity=serial.PARITY_NONE) 
	ser.open() #開く(ポートを開閉→マイコン側の再起動にも繋がるので注意)
	time.sleep(4) #Arduino側の再起動待機のため4秒停止(これは2~3秒もしくはそれ以下でも良いかも？(試して決定))
	ser.reset_input_buffer()#受信バッファのクリア
	
	#★
		
	#byteSendDataをフォーマット
	byteSendData =['s'.encode('utf-8')] 
	byteSendData.append(x.to_bytes(2,'big'))	#ビッグエンディアンでxをbyte型に変換＆リストに挿入
	byteSendData.append('/'.encode('utf-8'))
	byteSendData.append(y.to_bytes(2,'big'))	#ビッグエンディアンでyをbyte型に変換＆リストに挿入
	
	for b in byteSendData:
		ser.write(b)#送信
		ser.flush()#送信完了まで待機
		time.sleep(0.01)#安定のための待機

	try :
		byteReveiveData = ser.read(1)#1byte受信(1行読み込むならreadline)
		reseiveData = int.from_bytes(byteReveiveData, 'big')#byte型に変換
		ser.close()
		return reseiveData
	except ser.timeout as e: #タイムアウトエラー
		ser.close()
		raise e