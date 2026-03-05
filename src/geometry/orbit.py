from utils.constants import DEG_PER_CIRCLE

def roll_angle_deg(time_s: float, orbit_total_duration_s: float) -> float:
    phase = (time_s / orbit_total_duration_s) % 1.0
    return DEG_PER_CIRCLE * phase