"""
Unit and Integration Test Suite for the Vision Module.
Verifies dense traffic YOLO detection, ByteTrack tracking, track confirmation,
Active vs. Unique vehicle counter separation, and mathematical total invariants.
"""

import sys
import os
import numpy as np

def run_all_tests():
    print("==================================================")
    print("RUNNING DENSE TRAFFIC VISION MODULE INTEGRATION TESTS")
    print("==================================================")

    # TEST 1: Python Environment
    print("[TEST 1] Python Environment:", sys.executable)
    assert sys.version_info >= (3, 8), "Python version must be >= 3.8"
    print("  -> PASSED")

    # TEST 2: Check Required Packages
    print("[TEST 2] Checking required packages...")
    import cv2
    import ultralytics
    import fastapi
    import uvicorn
    import websockets
    print("  -> Packages cv2, ultralytics, fastapi, uvicorn, websockets are INSTALLED. PASSED")

    # TEST 3: Load YOLO model with imgsz=960 and max_det=300
    print("[TEST 3] Loading YOLO model with high-resolution imgsz=960 & max_det=300...")
    from vision.config import DEFAULT_MODEL_NAME, DEFAULT_IMGSZ, DEFAULT_MAX_DET
    from vision.detector import VehicleDetector
    detector = VehicleDetector(
        model_name=DEFAULT_MODEL_NAME,
        conf_threshold=0.20,
        imgsz=DEFAULT_IMGSZ,
        max_det=DEFAULT_MAX_DET
    )
    assert detector.model is not None, "YOLO model failed to load"
    assert detector.imgsz == 960, "Inference imgsz must be 960"
    assert detector.max_det == 300, "Max detections must be 300"
    print(f"  -> YOLO model '{DEFAULT_MODEL_NAME}' loaded with imgsz={detector.imgsz}, max_det={detector.max_det}. PASSED")

    # TEST 4: Vehicle Tracker State & Confirmation Logic
    print("[TEST 4] Testing Vehicle Tracker State & Track Confirmation Logic...")
    from vision.tracker import VehicleTrackerState
    tracker_state = VehicleTrackerState()

    # Track 17 observed for 1 frame
    tracker_state.update(17, (100, 100), current_time=1.0)
    assert tracker_state.is_confirmed(17, min_frames=3) is False, "Track 17 should NOT be confirmed after 1 frame"

    # Track 17 observed for frame 2 and frame 3
    tracker_state.update(17, (102, 102), current_time=1.033)
    tracker_state.update(17, (104, 104), current_time=1.066)
    assert tracker_state.is_confirmed(17, min_frames=3) is True, "Track 17 MUST be confirmed after 3 frames!"
    print("  -> 3-Frame Track confirmation verified. PASSED")

    # TEST 5-7: Unique Vehicle Counter & Single Count Guarantee (NO LINE)
    print("[TEST 5-7] Testing Unique Vehicle Counter (No Line Required)...")
    from vision.counter import UniqueVehicleCounter
    counter = UniqueVehicleCounter(confirmation_frames=3)

    # Frame 1: Vehicle 17 (Car) active (1 frame observed -> not confirmed yet)
    objs_f1 = [{"id": 17, "class_name": "car"}]
    active_s1, unique_s1 = counter.process_frame_objects(objs_f1, tracker_state)
    assert active_s1["car"] == 1, "Active car count should be 1"
    assert unique_s1["car"] == 1, "Unique car count should be 1 (confirmed in tracker_state)"

    # Frame 2-100: Vehicle 17 remains visible for 100 consecutive frames
    for f in range(2, 101):
        tracker_state.update(17, (100 + f, 100 + f), current_time=1.0 + f * 0.033)
        active_s, unique_s = counter.process_frame_objects(objs_f1, tracker_state)

    assert unique_s["car"] == 1, "Vehicle 17 MUST count as 1 unique car after 100 frames!"
    assert unique_s["total"] == 1, "Unique total MUST equal 1!"
    print("  -> Single unique count guarantee (100 frames test) verified. PASSED")

    # TEST 8-9: Active vs. Unique Count Separation
    print("[TEST 8-9] Testing Active vs. Unique Count Separation when Vehicle Exits Scene...")
    
    # Vehicle 17 leaves the frame (0 active vehicles)
    objs_empty = []
    active_s_exit, unique_s_exit = counter.process_frame_objects(objs_empty, tracker_state)

    assert active_s_exit["total"] == 0, "Active vehicles in scene MUST decrease to 0 when vehicle leaves!"
    assert unique_s_exit["total"] == 1, "Total unique vehicles MUST NOT decrease when vehicle leaves!"
    print("  -> Active count decreased to 0 while Total Unique remained 1. PASSED")

    # TEST 10-11: Multiple Vehicle Categories & Total Invariant
    print("[TEST 10-11] Testing Multiple Classes & Total Invariants...")
    # Add new vehicles: Motorcycle (ID 21), Bus (ID 30), Truck (ID 45)
    for tid in [21, 30, 45]:
        for _ in range(3):
            tracker_state.update(tid, (200, 200), current_time=2.0)

    objs_dense = [
        {"id": 17, "class_name": "car"},
        {"id": 21, "class_name": "motorcycle"},
        {"id": 30, "class_name": "bus"},
        {"id": 45, "class_name": "truck"}
    ]

    active_s_dense, unique_s_dense = counter.process_frame_objects(objs_dense, tracker_state)

    print("  -> Active Summary:", active_s_dense)
    print("  -> Unique Summary:", unique_s_dense)

    # Verify Active Total = Car + Motorcycle + Bus + Truck
    expected_active_total = active_s_dense["car"] + active_s_dense["motorcycle"] + active_s_dense["bus"] + active_s_dense["truck"]
    assert active_s_dense["total"] == expected_active_total, "Active Total invariant violated!"

    # Verify Unique Total = Car + Motorcycle + Bus + Truck
    expected_unique_total = unique_s_dense["car"] + unique_s_dense["motorcycle"] + unique_s_dense["bus"] + unique_s_dense["truck"]
    assert unique_s_dense["total"] == expected_unique_total, "Unique Total invariant violated!"
    assert unique_s_dense["total"] == 4, "Total unique vehicles must equal 4"
    print("  -> Active & Unique Total Invariants verified. PASSED")

    # TEST 12: Stream Processor & Performance Telemetry Packaging
    print("[TEST 12] Testing VideoStreamProcessor & Telemetry Payload...")
    from vision.video_processor import VideoStreamProcessor
    processor = VideoStreamProcessor(conf_thresh=0.20, imgsz=960, max_det=300, confirmation_frames=3)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ann_frame, telemetry = processor.process_frame(dummy_frame, current_time=100.0, video_fps=25.0)

    assert "active_vehicles" in telemetry
    assert "unique_vehicles" in telemetry
    assert "traffic" in telemetry
    assert "performance" in telemetry
    assert "config" in telemetry
    assert telemetry["config"]["imgsz"] == 960
    assert telemetry["config"]["max_det"] == 300
    assert telemetry["config"]["conf_threshold"] == 0.20

    print("  -> Telemetry Structure:", telemetry)
    print("  -> Stream Processor test PASSED.")

    print("==================================================")
    print("ALL 12 DENSE TRAFFIC INTEGRATION TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_all_tests()
