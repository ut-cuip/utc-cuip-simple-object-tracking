import datetime
import multiprocessing
import time

import cv2
import numpy as np

from pytorch_yolo.yolo import YOLO
from utils.sort import Sort
from utils.utils import distance, horizontal_distance, midpoint, vertical_distance


def cap_worker(cap_queue):
    tally = 0
    while tally < 1332:
        frame = cv2.imread("./frames/{}.jpg".format(tally))
        frame = frame[128 : frame.shape[0], 0 : frame.shape[1]]
        cap_queue.put(frame, block=True)
        tally += 1
    cap_queue.put(-1, block=True)


def main(cap_queue):
    yolo = YOLO()
    sort = Sort(iou_threshold=0.05)
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
    tally = 0
    while tally < 1332:
        frame = cap_queue.get(block=True)
        if type(frame) == int and frame == -1:
            return
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
                alive, _ = sort.update(detections, labels)
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
                    for trk2 in alive:
                        if trk == trk2:
                            continue
                        t2 = trk2.get_state()[0]
                        try:
                            bbox2 = [int(t2[0]), int(t2[1]), int(t2[2]), int(t2[3])]
                        except ValueError:
                            continue

                        d = distance(bbox, bbox2)
                        color = (255, 255, 255) if d > 50 else (0, 255, 255)

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
                        del d

                del t, bbox

        cv2.imshow("Object Tracking", frame)
        cv2.imwrite("./tracked/{}.png".format(tally), frame)
        tally += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    cap_queue = multiprocessing.Queue(1)
    processes = []
    processes.append(multiprocessing.Process(target=main, args=(cap_queue,)))
    processes.append(multiprocessing.Process(target=cap_worker, args=(cap_queue,)))

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
