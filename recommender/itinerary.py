"""
itinerary.py - Constraint-based (knowledge-based) itinerary planner.

Input : ranked places from the hybrid recommender + the traveller's constraints.
Output: a day-by-day plan.

Constraints (the "knowledge base"):
  * the place must be open when we arrive (we wait at most 1 hour for it to open)
  * the visit must finish before the place closes and before the day ends
  * the day's spending must stay within the daily budget
  * at most N stops per day (depends on pace)
  * a place is visited at most once in the whole trip

Algorithm (greedy, repeated for each day):
  start at the city centre at the day's start time
  repeat:
      for every unused place that satisfies all constraints:
          utility = recommendation score - DISTANCE_PENALTY * km to get there
      go to the place with the highest utility, advance the clock
  until nothing fits; then start the next day.
A 45-minute lunch break is inserted once the clock passes 1 pm.
"""

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

import pandas as pd

from .data import CITY_CENTRES

PACE_STOPS = {"Relaxed": 3, "Balanced": 4, "Packed": 6}
DISTANCE_PENALTY = 0.02      # score lost per km travelled
ROAD_FACTOR = 1.3            # roads are ~30% longer than straight lines
CITY_SPEED_KMPH = 20         # average city traffic speed
BUFFER_MIN = 10              # parking / finding transport
MAX_WAIT_HRS = 1.0
LUNCH_AFTER, LUNCH_LATEST, LUNCH_HRS = 13.0, 15.0, 0.75


@dataclass
class Stop:
    kind: str              # "visit" or "lunch"
    name: str
    arrive: float          # hours, e.g. 9.5 = 09:30
    depart: float
    place_id: str = ""
    category: str = ""
    travel_km: float = 0.0
    travel_min: float = 0.0
    cost: float = 0.0
    score: float = 0.0
    reason: str = ""
    latitude: float = 0.0
    longitude: float = 0.0


def haversine_km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def travel_minutes(km: float) -> float:
    return BUFFER_MIN + (km * ROAD_FACTOR) / CITY_SPEED_KMPH * 60


def fmt_time(hours: float) -> str:
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h:02d}:{m:02d}"


def plan_day(candidates: pd.DataFrame, used: set, city: str, start_hour: float,
             end_hour: float, daily_budget: float, max_stops: int,
             lunch: bool = True) -> list:
    pos = CITY_CENTRES[city]
    clock, spent, visits = start_hour, 0.0, 0
    lunch_done = not lunch
    stops = []

    while visits < max_stops:
        if not lunch_done and clock >= LUNCH_AFTER:
            if clock <= LUNCH_LATEST and clock + LUNCH_HRS <= end_hour:
                stops.append(Stop("lunch", "Lunch break", clock, clock + LUNCH_HRS))
                clock += LUNCH_HRS
            lunch_done = True

        best, best_utility = None, float("-inf")
        for row in candidates.itertuples():
            if row.place_id in used:
                continue
            km = haversine_km(pos, (row.latitude, row.longitude))
            t_min = travel_minutes(km)
            arrive = clock + t_min / 60
            if arrive < row.open_hour:
                if row.open_hour - arrive > MAX_WAIT_HRS:
                    continue
                arrive = row.open_hour
            depart = arrive + row.duration_hrs
            if depart > row.close_hour or depart > end_hour:
                continue
            if spent + row.cost_inr > daily_budget:
                continue
            utility = row.score - DISTANCE_PENALTY * km
            if utility > best_utility:
                best_utility = utility
                best = (row, km, t_min, arrive, depart)

        if best is None:
            break
        row, km, t_min, arrive, depart = best
        stops.append(Stop("visit", row.name, arrive, depart, row.place_id, row.category,
                          round(km, 1), round(t_min), row.cost_inr, round(row.score, 3),
                          row.reason, row.latitude, row.longitude))
        used.add(row.place_id)
        pos, clock, spent, visits = (row.latitude, row.longitude), depart, spent + row.cost_inr, visits + 1

    return stops


def plan_trip(recommendations: pd.DataFrame, city: str, days: int = 2,
              start_hour: float = 9.0, end_hour: float = 20.0,
              daily_budget: float = 2000, pace: str = "Balanced",
              lunch: bool = True) -> list:
    """Returns a list with one list of Stop objects per day."""
    candidates = recommendations[recommendations["city"] == city]
    used = set()
    trip = []
    for _ in range(days):
        trip.append(plan_day(candidates, used, city, start_hour, end_hour,
                             daily_budget, PACE_STOPS[pace], lunch))
    return trip


def trip_to_frame(trip: list) -> pd.DataFrame:
    rows = []
    for d, day in enumerate(trip, start=1):
        for s in day:
            rows.append({
                "day": d, "kind": s.kind, "time": f"{fmt_time(s.arrive)}-{fmt_time(s.depart)}",
                "place": s.name, "place_id": s.place_id, "category": s.category,
                "travel_km": s.travel_km, "travel_min": s.travel_min, "cost_inr": s.cost,
                "score": s.score, "why": s.reason,
                "latitude": s.latitude, "longitude": s.longitude,
            })
    return pd.DataFrame(rows)
