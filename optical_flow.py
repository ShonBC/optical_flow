import numpy as np
import cv2

def draw_vectors(frame, mag, ang):

    grid_size = int(25 / 2)



    # for k in range(mag):

    #     x = mag[k] * np.cos(ang[k])
    #     y = mag[k] * np.sin(ang[k])

    for i in range(len(frame)):
        for j in range(len(frame[i])):

            x = j * grid_size
            y = i * grid_size

            x_start ,x_end, y_start,y_end = arrows(mag, ang)

            cv2.circle(frame, (x, y), 1, (255, 255, 255), 1)
            cv2.arrowedLine(frame, (x, y), )

def arrows(mag, ang):

    x = mag * np.cos(ang)
    y = mag * np.sin(ang)

    x_start ,x_end, y_start,y_end = x[0], x[1], y[0], y[1]

    return x_start ,x_end, y_start,y_end


def flow_filter():

    cap = cv2.VideoCapture(cv2.samples.findFile("Cars On Highway.mp4"))
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while True:
        
        ret, frame2 = cap.read()

        if not ret:
            break

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 25, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        draw_vectors(frame2, mag, ang)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2', bgr)
        cv2.imshow('flow', frame2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', bgr)
        prvs = next

if __name__ == "__main__":

    flow_filter() 