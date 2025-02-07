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
    src_center_offset = [94, 77]  #121, 42 94, 77 80, 96

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

def draw(bin_img):
    histogram = np.sum(bin_img, axis=0)
    midpoint = np.int(histogram.shape[0]/2) # 160
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint
    # print(leftbase)
    # print(rightbase)
    # plt.plot(histogram)
    # plt.show()

    out_img = np.dstack((bin_img, bin_img, bin_img)) * 255

    ## window parameter
    # 적절한 윈도우의 개수를 지정합니다.
    nwindows = 7
    # 개수가 너무 적으면 정확하게 차선을 찾기 힘듭니다.
    # 개수가 너무 많으면 연산량이 증가하여 시간이 오래 걸립니다.
    window_height = 240//nwindows
    # 윈도우의 너비를 지정합니다. 윈도우가 옆 차선까지 넘어가지 않게 사이즈를 적절히 지정합니다.
    margin = 25
    # 탐색할 최소 픽셀의 개수를 지정합니다.
    min_pix = round((margin * 2 * window_height) * 0.0031)

    lane_pixel = bin_img.nonzero()
    lane_pixel_y = np.array(lane_pixel[0])
    lane_pixel_x = np.array(lane_pixel[1])

    # pixel index를 담을 list를 만들어 줍니다.
    left_lane_idx = []
    right_lane_idx = []

    for window in range(nwindows): # 1~n 반복
        win_y_low = bin_img.shape[0] - (window + 1) * window_height # 윈도우 아래 인덱스
        win_y_high = bin_img.shape[0] - window * window_height # 윈도우 위 인덱스

        # 윈도우 넓이
        win_x_left_low = leftbase - margin 
        win_x_left_high = leftbase + margin
        win_x_right_low = rightbase - margin
        win_x_right_high = rightbase + margin

        if leftbase != 0:
            cv2.rectangle(
                out_img,
                (win_x_left_low, win_y_low),
                (win_x_left_high, win_y_high),
                (0, 255, 0),2)
        if rightbase != midpoint:
            cv2.rectangle(out_img,
                (win_x_right_low, win_y_low),
                (win_x_right_high, win_y_high),
                (0, 0, 255),2)

        # 왼쪽 오른쪽 각 차선 픽셀이 window안에 있는 경우 index를 저장합니다.
        good_left_idx = (
            (lane_pixel_y >= win_y_low) & (lane_pixel_y < win_y_high) & (lane_pixel_x >= win_x_left_low) & (lane_pixel_x < win_x_left_high)
        ).nonzero()[0]
        good_right_idx = (
            (lane_pixel_y >= win_y_low) & (lane_pixel_y < win_y_high) & (lane_pixel_x >= win_x_right_low) & (lane_pixel_x < win_x_right_high)
        ).nonzero()[0]

        left_lane_idx.append(good_left_idx)
        right_lane_idx.append(good_right_idx)

        # window내 설정한 pixel개수 이상이 탐지되면, 픽셀들의 x 좌표 평균으로 업데이트 합니다.
        if len(good_left_idx) > min_pix:
            leftbase = np.int32(np.mean(lane_pixel_x[good_left_idx]))
        if len(good_right_idx) > min_pix:
            rightbase = np.int32(np.mean(lane_pixel_x[good_right_idx]))


    # np.concatenate(array) => axis 0으로 차원 감소 시킵니다.(window개수로 감소)
    left_lane_idx = np.concatenate(left_lane_idx)
    right_lane_idx = np.concatenate(right_lane_idx)

    # window 별 좌우 도로 픽셀 좌표입니다.
    left_x = lane_pixel_x[left_lane_idx]
    left_y = lane_pixel_y[left_lane_idx]
    right_x = lane_pixel_x[right_lane_idx]
    right_y = lane_pixel_y[right_lane_idx]

    # 좌우 차선 별 2차함수 계수를 추정합니다.
    if len(left_x) < 700:
        left_x = right_x - 230
        left_y = right_y
    elif len(right_x) < 700:
        right_x = left_x + 230
        right_y = left_y

    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    # 좌우 차선 별 추정할 y좌표입니다.
    plot_y = np.linspace(0, bin_img.shape[0] - 1, nwindows)
    # 좌우 차선 별 2차 곡선을 추정합니다.
    left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
    center_fit_x = (right_fit_x + left_fit_x) / 2

    # 양쪽 차선 및 중심 선 pixel 좌표(x,y)로 변환합니다.
    center = np.asarray(tuple(zip(center_fit_x, plot_y)), np.int32)
    right = np.asarray(tuple(zip(right_fit_x, plot_y)), np.int32)
    left = np.asarray(tuple(zip(left_fit_x, plot_y)), np.int32)

    cv2.polylines(out_img, [left], False, (0, 0, 255), thickness=5)
    cv2.polylines(out_img, [right], False, (0, 255, 0), thickness=5)
    sliding_window_img = out_img
    return sliding_window_img, left, right, center, left_x, left_y, right_x, right_y

cap = cv2.VideoCapture('21-09-17-12-39-53.mp4')
while True:
    ret, img = cap.read()
    if not ret:
        break
    filtered_img = color(img)
    warp_img = warp(filtered_img)
    bin_img = binary(warp_img)
    sliding_window_img,left,right,center,left_x,_,right_x,_ = draw(bin_img)
    
    # print("left-----------")
    # print(np.average(left_x))
    # print(left)
    # print('right--------')
    # print(np.average(right_x))
    # print(right)
    # print([len(right_x),len(left_x)])
    # print(center[:,0])

    
    print("-----------")
    if np.average(center[:,0])> 160:
        print("turn right")
    else:
        print("turn left")


    cv2.imshow("filtered_img", filtered_img)
    # cv2.imshow("warp_img", warp_img)
    cv2.imshow("bin_img", bin_img)
    cv2.imshow("out_img", sliding_window_img)


    key = cv2.waitKey(1)
    if  key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

