# -*- coding: utf-8 -*-

from serialPackage import armCommunication
if __name__ == '__main__': 
	armUSB = armCommunication('/dev/tty.usbmodem14101',115200,0.01);
	armUSB.isWritingXY(100,120)
	while True:
		receiveData = armUSB.isReadingOneByte()
		if	receiveData==10:
			print("NO")
		elif	receiveData==11:
			print("OK")