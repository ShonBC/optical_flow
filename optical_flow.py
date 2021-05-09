import numpy as np
import cv2

def vector_field():

    cap = cv2.VideoCapture(cv2.samples.findFile("Cars On Highway.mp4"))
    ret, frame1 = cap.read()
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while True:
        
        ret, frame2 = cap.read()

        if not ret:
            break

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next_gray, None, 0.5, 3, 25, 3, 5, 1.2, 0)

        draw_vec_field(frame2, next_gray, flow)
            
        cv2.imshow('flow', frame2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            # cv2.imwrite('opticalhsv.png', bgr)
        prev = next_gray


def draw_vec_field(frame, img, flow, grid_size=25):

    height, width = img.shape

    # Create Grid (Default segments image into square sections with size 25 by 25 pixels)
    y, x = np.mgrid[grid_size / 2 : height : grid_size, grid_size / 2 : width : grid_size].reshape(2,-1).astype(int)

    # Define flow vectors
    fx, fy = flow[y,x].T
    vec = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    vec = np.int32(vec)

    # Draw flow vecotrs with origens at center grid sections)
    cv2.polylines(frame, vec, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in vec:
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)


def flow_filter():

    cap = cv2.VideoCapture(cv2.samples.findFile("Cars On Highway.mp4"))
    ret, frame1 = cap.read()
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    motion = np.zeros_like(frame1)

    while True:
        
        ret, frame2 = cap.read()

        if not ret:
            break

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next_gray, None, 0.5, 3, 25, 3, 5, 1.2, 0)

        # Flow Vecotrs
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Show pixels with motion in HSV color space
        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = norm_mag
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # If the normalized magnitude of the optical flow vector is greater than 25, set the mask image intensity equal to the original frame intensity.
        for i in range(len(norm_mag)):
            for j in range(len(norm_mag[i])):

                if norm_mag[i][j] > 25:
                    motion[i][j] = frame2[i][j]

        cv2.imshow('Color Coded Optical Flow', bgr)
        cv2.imshow('Filtered Motion', motion)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', bgr)
        prev = next_gray

if __name__ == "__main__":

    flow_filter() 
    # vector_field()