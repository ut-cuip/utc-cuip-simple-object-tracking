import multiprocessing
import time

import cv2
import numpy as np

from pytorch_yolo.yolo import YOLO
from utils.sort import Sort


def cap_worker(queue):
    while True:
        cap = cv2.VideoCapture()
        cap.open(
            "rtsp://consumer:IyKv4uY7%g^8@10.199.51.162/axis-media/media.amp?streamprofile=H264-OpenCV-Optimized")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while True:
            ret, frame = cap.read()
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            if not ret:
                cap.release()
                del cap
                time.sleep(10)
                break
            else:
                queue.put(frame, block=True)


def main(queue):
    yolo = YOLO()
    sort = Sort()
    whitelist = [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "truck",
        "boat",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bird",
        "cat",
        "dog",
        "backpack",
        "umbrella",
        "handbag",
        "suitcase",
    ]
    while True:
        frame = queue.get(block=True)
        outputs = yolo.get_results(frame)
        detections = []
        labels = []

        for output in outputs:
            label = yolo.classes[int(output[-1])]
            if not label in whitelist:
                del label
                continue

            tl = tuple(output[1:3].int())
            br = tuple(output[3:5].int())

            detections.append(
                np.array(
                    [tl[0].item(), tl[1].item(), br[0].item(), br[1].item()]
                )
            )
            labels.append(label)

            del tl, br, label
        del outputs

        detections = np.array(detections)

        if detections.shape[0] > 0:
            try:
                alive, dead = sort.update(detections, labels)
            except IndexError:
                del frame, detections, labels
                continue

            for trk in alive:
                t = trk.get_state()[0]
                try:
                    bbox = [int(t[0]), int(t[1]), int(t[2]), int(t[3])]
                except ValueError:
                    continue
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    trk.color,
                    2,
                )
                cv2.putText(
                    frame,
                    "{}:id {}".format(trk.get_label(), str(trk.id)),
                    (int(bbox[0]), int(bbox[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.0005 * frame.shape[0],
                    trk.color,
                    2,
                )
                for location in trk.locations:
                    x1, y1, x2, y2 = location[1]
                    cv2.circle(frame, (((int(x1) + int(x2))//2),
                                       int(y2)), 3, trk.color, -1)
                    del x1, y1, x2, y2

                del t, bbox
            cv2.imshow("Object Tracking", frame)
            key = cv2.waitKey(1) & 0XFF
            if key == ord('q'):
                exit()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    queue = multiprocessing.Queue(32)
    processes = []
    processes.append(multiprocessing.Process(target=main, args=(queue,)))
    processes.append(multiprocessing.Process(target=cap_worker, args=(queue,)))

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
