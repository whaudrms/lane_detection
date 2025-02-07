import cv2
import numpy as np
import matplotlib.pyplot as plt

def color(img):  #하얀선과 노란선을 추출하여 띄우기
    # -> HSV 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    white_lower = np.array([0, 0, 200])
    white_upper = np.array([179, 255, 255])
    white_range = cv2.inRange(hsv, white_lower, white_upper)
    filtered_img = cv2.bitwise_and(img, img, mask=white_range)
    return filtered_img

def warp(img): #이미지를 버드와이저로 변환
    x, y = img.shape[1], img.shape[0] #320,240

    # ROi
    src_center_offset = [115, 94]  #121, 42 94, 77

    src_point1=[0, 240]
    src_point2=[src_center_offset[0], src_center_offset[1]] 
    src_point3=[320 - src_center_offset[0], src_center_offset[1]] 
    src_point4=[320, 240]
    src=np.float32([src_point1,src_point2,src_point3,src_point4])
    
    dst_offset = [round(x * 0.20), 0]
        # offset x 값이 작아질 수록 dst box width 증가합니다.

    dst_point1=[dst_offset[0], y]
    dst_point2=[dst_offset[0], 0]
    dst_point3=[x - dst_offset[0], 0]
    dst_point4=[x - dst_offset[0], y]
    dst=np.float32([dst_point1,dst_point2,dst_point3,dst_point4])
    
    # find perspective matrix
    matrix = cv2.getPerspectiveTransform(src, dst)
    warp_img = cv2.warpPerspective(img, matrix, (x, y))
    return warp_img

def binary(warp_img): #bin으로 변환
    grayed_img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(grayed_img, 160, 255, cv2.THRESH_BINARY)
    return bin_img

def stop_line(bin_img,warp_img):
    # 확률적 허프 변환 수행
    rho = 1                   # 허프 그리드에서의 거리 해상도 (픽셀 단위)
    theta = np.pi / 180       # 허프 그리드에서의 각도 해상도 (라디안 단위)
    threshold = 20           # 선으로 간주되기 위한 최소 교차점 수
    min_line_length = 50     # 선의 최소 길이
    max_line_gap = 100        # 선분 사이의 최대 허용 간격
    
    # edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)
    stop_lines = cv2.HoughLinesP(bin_img, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if stop_lines is not None:
        for line in stop_lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # 수직 방향으로 흐르는 것을 고려하여 더 작은 값으로 설정
                cv2.line(warp_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
                print('stop_line')
    
cap = cv2.VideoCapture('KakaoTalk_20240519_210000257.mp4')
while True:
    ret, img = cap.read()
    if not ret:
        break

    filtered_img = color(img)
    warp_img = warp(filtered_img)
    bin_img = binary(warp_img)
    stop_line(bin_img,warp_img) 
    
    # cv2.imshow("filtered_img", filtered_img)
    # cv2.imshow("bin_img", bin_img)
    cv2.imshow("warp_img", warp_img)


    key = cv2.waitKey(20)
    if  key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

