"""
Unique Vehicle Counter module.
Maintains persistent unique track ID confirmation counting.
Separates ACTIVE VEHICLES (in current frame) from TOTAL UNIQUE VEHICLES DETECTED (cumulative session).
"""

import logging
from typing import Dict, Set, List, Tuple
from vision.config import DEFAULT_CONFIRMATION_FRAMES

logger = logging.getLogger(__name__)


class UniqueVehicleCounter:
    """Manages Active Vehicle counts and confirmed Cumulative Unique Vehicle counts."""

    def __init__(self, confirmation_frames: int = DEFAULT_CONFIRMATION_FRAMES):
        self.confirmation_frames = confirmation_frames

        # Cumulative confirmed unique vehicle counts
        self.unique_counts: Dict[str, int] = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }

        # Set of vehicle track IDs confirmed as unique
        self.confirmed_unique_ids: Set[int] = set()

        # Map of track ID -> class_name for confirmed tracks
        self.confirmed_class_map: Dict[int, str] = {}

    def update_confirmation_frames(self, frames: int) -> None:
        """Update required confirmation frame count live."""
        self.confirmation_frames = max(1, min(10, frames))

    def process_frame_objects(self, tracked_objects: List[Dict], tracker_state) -> Tuple[Dict, Dict]:
        """
        Process active tracked objects for the current frame.

        Args:
            tracked_objects: List of dicts representing detected objects in current frame.
            tracker_state: VehicleTrackerState instance tracking observation counts.

        Returns:
            Tuple of (active_summary_dict, unique_summary_dict)
        """
        # Reset active frame counters
        active_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }

        for obj in tracked_objects:
            track_id = obj.get("id", -1)
            cls_name = obj.get("class_name", "car")

            if cls_name not in active_counts:
                cls_name = "car"

            # Increment active vehicle count for this frame
            active_counts[cls_name] += 1

            # Check if this track is confirmed as a valid unique vehicle
            if track_id >= 0:
                is_confirmed = tracker_state.is_confirmed(track_id, self.confirmation_frames)
                obj["is_confirmed"] = is_confirmed

                # If confirmed and not yet in confirmed_unique_ids set, count it ONCE
                if is_confirmed and track_id not in self.confirmed_unique_ids:
                    self.confirmed_unique_ids.add(track_id)
                    self.confirmed_class_map[track_id] = cls_name
                    self.unique_counts[cls_name] += 1
                    logger.info(
                        f"NEW UNIQUE VEHICLE CONFIRMED: ID {track_id} ({cls_name}). "
                        f"Total Unique: {self.get_total_unique_count()}"
                    )
            else:
                obj["is_confirmed"] = False

        # Build summaries enforcing strict Total = Car + Motorcycle + Bus + Truck invariant
        active_summary = {
            "car": active_counts["car"],
            "motorcycle": active_counts["motorcycle"],
            "bus": active_counts["bus"],
            "truck": active_counts["truck"],
            "total": sum(active_counts.values())
        }

        unique_summary = {
            "car": self.unique_counts["car"],
            "motorcycle": self.unique_counts["motorcycle"],
            "bus": self.unique_counts["bus"],
            "truck": self.unique_counts["truck"],
            "total": sum(self.unique_counts.values())
        }

        return active_summary, unique_summary

    def get_total_unique_count(self) -> int:
        """Get cumulative unique count."""
        return sum(self.unique_counts.values())

    def reset(self) -> None:
        """Reset all unique counts and confirmed track ID set."""
        for k in self.unique_counts:
            self.unique_counts[k] = 0
        self.confirmed_unique_ids.clear()
        self.confirmed_class_map.clear()
        logger.info("Unique vehicle counter reset to zero.")
