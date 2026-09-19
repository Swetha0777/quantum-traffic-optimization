"""
Emergency Corridor Management

Purpose:
    Detect and manage an emergency vehicle corridor
    across the traffic network.

Network:
    J1 -> J2 -> J4

The module does NOT perform QAOA.
It creates emergency constraints/priority information
which can be used together with the QAOA result.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


# ============================================================
# DEFAULT EMERGENCY ROUTE
# ============================================================

DEFAULT_EMERGENCY_ROUTE = [
    "J1",
    "J2",
    "J4"
]


# ============================================================
# EMERGENCY VEHICLE
# ============================================================

@dataclass
class EmergencyVehicle:

    vehicle_id: str = "EV-01"

    route: Optional[List[str]] = None

    current_index: int = 0

    active: bool = True

    # Estimated speed in km/h
    speed_kmh: float = 40.0

    def __post_init__(self):

        if self.route is None:
            self.route = DEFAULT_EMERGENCY_ROUTE.copy()

    # --------------------------------------------------------
    # CURRENT INTERSECTION
    # --------------------------------------------------------

    @property
    def current_intersection(self):

        if not self.active:
            return None

        if self.current_index >= len(self.route):
            return None

        return self.route[
            self.current_index
        ]

    # --------------------------------------------------------
    # NEXT INTERSECTION
    # --------------------------------------------------------

    @property
    def next_intersection(self):

        next_index = self.current_index + 1

        if next_index >= len(self.route):
            return None

        return self.route[
            next_index
        ]

    # --------------------------------------------------------
    # COMPLETED?
    # --------------------------------------------------------

    @property
    def completed(self):

        return (
            not self.active
            and
            self.current_index >= len(self.route)
        )

    # --------------------------------------------------------
    # MOVE VEHICLE
    # --------------------------------------------------------

    def advance(self):

        if not self.active:
            return self.status()

        self.current_index += 1

        if self.current_index >= len(self.route):

            self.active = False

        return self.status()

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def status(self):

        return {
            "vehicle_id": self.vehicle_id,
            "active": self.active,
            "current_intersection":
                self.current_intersection,
            "next_intersection":
                self.next_intersection,
            "route":
                self.route,
            "current_index":
                self.current_index
        }


# ============================================================
# EMERGENCY CORRIDOR
# ============================================================

class EmergencyCorridor:

    def __init__(
        self,
        route=None
    ):

        self.route = (
            route.copy()
            if route
            else DEFAULT_EMERGENCY_ROUTE.copy()
        )

    # --------------------------------------------------------
    # CHECK INTERSECTION
    # --------------------------------------------------------

    def is_in_corridor(
        self,
        intersection_id
    ):

        return (
            intersection_id
            in self.route
        )

    # --------------------------------------------------------
    # GET PRIORITY LEVEL
    # --------------------------------------------------------

    def get_priority(
        self,
        intersection_id,
        emergency_vehicle
    ):

        if not emergency_vehicle.active:

            return "NORMAL"

        current = (
            emergency_vehicle.current_intersection
        )

        next_intersection = (
            emergency_vehicle.next_intersection
        )

        if intersection_id == current:

            return "ACTIVE"

        if intersection_id == next_intersection:

            return "PREPARE"

        if intersection_id in self.route:

            return "CORRIDOR"

        return "NORMAL"

    # --------------------------------------------------------
    # MARK TRAFFIC DATA
    # --------------------------------------------------------

    def mark_traffic_data(
        self,
        traffic_data: pd.DataFrame,
        emergency_vehicle: EmergencyVehicle
    ):

        data = traffic_data.copy()

        # Make sure columns exist

        data["emergency_active"] = (
            emergency_vehicle.active
        )

        data["is_emergency_route"] = False

        data["emergency_priority"] = "NORMAL"

        data["emergency_current"] = False

        data["emergency_next"] = False

        for index, row in data.iterrows():

            intersection = str(
                row["intersection_id"]
            )

            priority = self.get_priority(
                intersection,
                emergency_vehicle
            )

            data.loc[
                index,
                "is_emergency_route"
            ] = (
                intersection
                in self.route
            )

            data.loc[
                index,
                "emergency_priority"
            ] = priority

            data.loc[
                index,
                "emergency_current"
            ] = (
                priority == "ACTIVE"
            )

            data.loc[
                index,
                "emergency_next"
            ] = (
                priority == "PREPARE"
            )

        return data

    # --------------------------------------------------------
    # CREATE SIGNAL OVERRIDE
    # --------------------------------------------------------

    def create_signal_override(
        self,
        traffic_data: pd.DataFrame,
        emergency_vehicle: EmergencyVehicle
    ):

        overrides = {}

        current = (
            emergency_vehicle.current_intersection
        )

        next_intersection = (
            emergency_vehicle.next_intersection
        )

        for _, row in traffic_data.iterrows():

            intersection = str(
                row["intersection_id"]
            )

            # --------------------------------------------
            # CURRENT EMERGENCY INTERSECTION
            # --------------------------------------------

            if intersection == current:

                overrides[intersection] = {

                    "signal":
                        "NS_GREEN",

                    "green_time":
                        60,

                    "priority":
                        "ACTIVE",

                    "reason":
                        "Emergency vehicle approaching"
                }

            # --------------------------------------------
            # NEXT INTERSECTION
            # --------------------------------------------

            elif intersection == next_intersection:

                overrides[intersection] = {

                    "signal":
                        "NS_GREEN",

                    "green_time":
                        50,

                    "priority":
                        "PREPARE",

                    "reason":
                        "Prepare emergency corridor"
                }

            # --------------------------------------------
            # OTHER CORRIDOR
            # --------------------------------------------

            elif intersection in self.route:

                overrides[intersection] = {

                    "signal":
                        "NS_GREEN",

                    "green_time":
                        40,

                    "priority":
                        "CORRIDOR",

                    "reason":
                        "Emergency route"
                }

            # --------------------------------------------
            # NORMAL TRAFFIC
            # --------------------------------------------

            else:

                overrides[intersection] = {

                    "signal":
                        str(
                            row.get(
                                "current_signal",
                                "NS_GREEN"
                            )
                        ),

                    "green_time":
                        30,

                    "priority":
                        "NORMAL",

                    "reason":
                        "Normal traffic"
                }

        return overrides

    # --------------------------------------------------------
    # ADVANCE EMERGENCY VEHICLE
    # --------------------------------------------------------

    def advance(
        self,
        emergency_vehicle
    ):

        return emergency_vehicle.advance()

    # --------------------------------------------------------
    # END EMERGENCY
    # --------------------------------------------------------

    def finish(
        self,
        emergency_vehicle
    ):

        emergency_vehicle.active = False

        emergency_vehicle.current_index = (
            len(emergency_vehicle.route)
        )

        return emergency_vehicle.status()