import numpy as np
import cv2

def draw_vectors():

    cap = cv2.VideoCapture(cv2.samples.findFile("Cars On Highway.mp4"))
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    grid_size = int(25 / 2)

    while True:
        
        ret, frame2 = cap.read()

        if not ret:
            break

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_gray, None, 0.5, 3, 25, 3, 5, 1.2, 0)

        draw_flow(frame2, next_gray, flow, step=16)
            
        cv2.imshow('flow', frame2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            # cv2.imwrite('opticalhsv.png', bgr)
        prvs = next_gray

def arrows(mag, ang):

    x = mag * np.cos(ang)
    y = mag * np.sin(ang)

    x_start ,x_end, y_start,y_end = x[0], x[1], y[0], y[1]

    return x_start ,x_end, y_start,y_end

def draw_flow(frame, img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    vec = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    vec = np.int32(vec + 0.5)

    cv2.polylines(frame, vec, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in vec:
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)


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

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_gray, None, 0.5, 3, 25, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # draw_vectors(frame2, mag, ang)
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
        prvs = next_gray

if __name__ == "__main__":

    # flow_filter() 
    draw_vectors()