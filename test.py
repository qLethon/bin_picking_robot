import cv2
import numpy as np
from PIL import Image

def transform_by4(img, points):
    """ copied from https://blanktar.jp/blog/2015/07/python-opencv-crop-box.html """
    """ 4点を指定してトリミングする。 """

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

    trans = cv2.getPerspectiveTransform(points, dst)  # 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。
    return cv2.warpPerspective(img, trans, (int(width), int(height)))  # 透視変換行列を使って切り抜く。

def transform_image(image):
    blue, green, red = cv2.split(image)
    diff = np.where(red >= green, red - (green.astype(np.uint16) * 3 // 10).astype(np.uint8), 0)
    cv2.imshow("diff", diff)
    ret, thresh = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("b", thresh)
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

capture(2).save('a.jpg')
transform_image(cv2.imread('a.jpg'))