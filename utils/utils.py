import math

def distance(bbox1, bbox2):
    a1, b1, a2, b2 = bbox1
    x1, y1, x2, y2 = bbox2
    center1 = (a1 + a2) // 2
    center2 = (x1 + x2) // 2
    return int(math.sqrt((abs(center1 - center2) ** 2) + (abs(b2 - y2) ** 2)))

def midpoint(bbox1, bbox2):
    a1, b1, a2, b2 = bbox1
    x1, y1, x2, y2 = bbox2
    center1 = (a1 + b1) // 2
    center2 = (x1 + x2) // 2
    return (((center1 + center2) // 2), ((b2 + y2) // 2))