import multiprocessing
import time

import cv2
import numpy as np

from pytorch_yolo.yolo import YOLO
from utils.sort import Sort
from utils.utils import midpoint, distance


def cap_worker(queue):
    while True:
        cap = cv2.VideoCapture()
        cap.open(
            "rtsp://consumer:IyKv4uY7%g^8@10.199.51.162/axis-media/media.amp?streamprofile=H264-OpenCV-Optimized"
        )
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while True:
            ret, frame = cap.read()
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
    frames = 0
    while True:
        frame = queue.get(block=True)
        frames += 1
        # if frames % 2 == 0:
        #     continue
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
                np.array([tl[0].item(), tl[1].item(), br[0].item(), br[1].item()])
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
                    cv2.circle(
                        frame, (((int(x1) + int(x2)) // 2), int(y2)), 3, trk.color, -1
                    )
                    del x1, y1, x2, y2
                if len(alive) > 1:
                    distances = []
                    for trk2 in alive:
                        if(trk == trk2):
                            continue
                        t2 = trk2.get_state()[0]
                        try:
                            bbox2 = [int(t2[0]), int(t2[1]), int(t2[2]), int(t2[3])]
                        except ValueError:
                            continue

                        d = distance(bbox, bbox2)
                        if d in distances:
                            continue
                        else:
                            distances.append(d)
                        threshold = 100
                        color = (255, 255, 255) if d > threshold else (0, 0, 255)
                        cv2.line(
                            frame,
                            ((bbox[0] + bbox[2]) // 2, int(bbox[3])),
                            ((bbox2[0] + bbox2[2]) // 2, int(bbox2[3])),
                            color,
                            2,
                        )
                        cv2.putText(
                            frame,
                            str(d),
                            midpoint(bbox, bbox2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.0005 * frame.shape[0],
                            color,
                            2,
                        )
                        del d, color, threshold
                    del distances

                del t, bbox
            cv2.imshow("Object Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    queue = multiprocessing.Queue(1)
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
