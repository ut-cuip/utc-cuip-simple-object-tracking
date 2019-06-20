from utils.utils import distance_from_bb, distance_from_xy
import datetime


def predict_location(trk, velocity, dt=1):
    """
        Predicts the future location at 'time'
        Args:
            trk (sort.KalmanBoxTracker): the tracker
            velocity (float): the velocity of trk
            dt (int): delta in time (in seconds)
    """
    timestamp, location = trk.locations[-1]
    pred_x = ((location[0] + location[2]) / 2) + (velocity * dt)
    pred_y = location[3] + (velocity * dt)
    return (pred_x, pred_y)


def velocity(trk):
    """
        Calculates the velocity of said tracker, in px/s
        Args:
            trk (sort.KalmanBoxTracker)
    """
    prev_timestamp, prev_hitbox = trk.locations[-2]
    current_timestamp, current_hitbox = trk.locations[-1]
    dt = current_timestamp - prev_timestamp

    vel_x = (
        ((prev_hitbox[0] + prev_hitbox[2]) / 2)
        - ((current_hitbox[0] + current_hitbox[2]) / 2)
    ) / dt

    vel_y = (prev_hitbox[3] - current_hitbox[3]) / dt

    return (vel_x + vel_y) / 2


def time_til_collision(trk1, trk2, threshold=30):
    """
        Determines if two trackers should collide
        Args:
            trk1 (sort.KalmanBoxTracker)
            trk2 (sort.KalmanBoxTracker)
        returns:
            -1 if no collision
            ttc if collision is possible
    """
    if len(trk1.get_state()) < 2 or len(trk2.get_state()) < 2:
        print("trk1 or trk2 had too few points to calculate velocity.")
        return -1

    vel_trk1 = velocity(trk1)
    vel_trk2 = velocity(trk2)
    known_distance = distance_from_bb(trk1.locations[-1], trk2.locations[-1])

    pred_trk1_pos = predict_location(trk1, vel_trk1)
    pred_trk2_pos = predict_location(trk2, vel_trk2)
    pred_distance = distance_from_xy(pred_trk1_pos, pred_trk2_pos)

    # If distance is decreasing...
    if known_distance > pred_distance:
        ttc = 0
        while known_distance + (pred_distance - known_distance) * ttc > 0:
            ttc += 1
        return ttc
    else:
        return -1
