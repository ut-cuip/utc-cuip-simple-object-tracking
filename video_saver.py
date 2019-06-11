import datetime

import cv2
import time

if __name__ == "__main__":
    cap = cv2.VideoCapture()
    cap.open(
        "rtsp://consumer:IyKv4uY7%g^8@10.199.51.162/axis-media/media.amp"
    )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    start_time = time.time()
    tally = 0
    while time.time() - start_time < 45:
        loop_start = time.time()
        ret, frame = cap.read()
        cv2.imwrite("frames/{}.jpg".format(tally), frame)
        tally += 1
        if ((1 / 30) - (time.time() - loop_start)) > 0:
            time.sleep((1 / 30) - (time.time() - loop_start))
    cap.release()

