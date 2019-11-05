import cv2
import numpy as np
from PIL import Image

def update_points(points):
    pointsOldDataFile = open('pointsOldData.csv','w')
    for _point in points:
        pointLineString = str(_point[0])+","+str(_point[1]) + "\n"
        pointsOldDataFile.write(pointLineString)
    pointsOldDataFile.close()

def read_savedPoints():
    points = np.array([])
    pointsOldDataFile = open('pointsOldData.csv','r')
    for pointLineString_fromFile in pointsOldDataFile.readlines():
        pointStrings = pointLineString_fromFile.split(",")
        pointFloat = [float(p) for p in pointStrings]
        np.insert(points,pointFloat,len(points), axis=0)
    pointsOldDataFile.close()
    return points

def transform_by4(img, points):
    """ copied from https://blanktar.jp/blog/2015/07/python-opencv-crop-box.html """
    """ 4点を指定してトリミングする。 """

    if(not len(points)==4): #頂点の数が4つでないなら古いデータを使う
        print("ないんじゃ～～")
        points = read_savedPoints()
    else:                   #頂点の数が4つなら古いデータ更新
        update_points(points)
    points = sorted(points, key=lambda x:x[1])  # yが小さいもの順に並び替え。
    top = sorted(points[:2], key=lambda x:x[0])  # 前半二つは四角形の上。xで並び替えると左右も分かる。
    bottom = sorted(points[2:], key=lambda x:x[0], reverse=True)  # 後半二つは四角形の下。同じくxで並び替え。
    points = np.array(top + bottom, dtype='float32')  # 分離した二つを再結合。
    width = max(np.sqrt(((points[0][0]-points[2][0])**2)*2), np.sqrt(((points[1][0]-points[3][0])**2)*2))
    height = max(np.sqrt(((points[0][1]-points[2][1])**2)*2), np.sqrt(((points[1][1]-points[3][1])**2)*2))

    dst = np.array([
            np.array([0, 0]),
            np.array([width-1, 0]),
            np.array([width-1, height-1]),
            np.array([0, height-1]),
            ], np.float32)
    trans = cv2.getPerspectiveTransform(points, dst)  # 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。(射影行列では？)
    return cv2.warpPerspective(img, trans, (int(width), int(height)))  #ここで影を指定のサイズで受け取る

def transform_image(image):
    blue, green, red = cv2.split(image)
    diff = np.where(green >= red, green - (red.astype(np.uint16) * 10 // 10).astype(np.uint8), 0)
    cv2.imshow("diff", diff)
    ret, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("b", thresh)
    kernel = np.ones((50,50),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("close", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    epsilon = 0.05 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    cv2.waitKey(0)
    cv2.drawContours(image, [approx], 0, (0,255,0), 3)
    cv2.imshow("a", image)
    
    cv2.waitKey(0)
    cv2.imwrite("tranformed.jpg", transform_by4(image, approx[:, 0, :]))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def capture(num):
    cam = cv2.VideoCapture(num)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 4000)
    retval, frame = cam.read()
    if not retval:
        print('cannnot read')
    cam.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#capture(2).save('a.jpg')
transform_image(cv2.imread('a.jpg'))