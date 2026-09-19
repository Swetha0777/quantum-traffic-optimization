/**
 * Frontend JavaScript Client for AI Traffic Detection & Track Confirmation Counter
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const connectionStatus = document.getElementById("connectionStatus");
  const statusText = document.getElementById("statusText");

  const btnUploadVideo = document.getElementById("btnUploadVideo");
  const videoFileInput = document.getElementById("videoFileInput");
  const btnStartCamera = document.getElementById("btnStartCamera");
  const btnStopStream = document.getElementById("btnStopStream");
  const btnResetCounters = document.getElementById("btnResetCounters");

  const confThreshSlider = document.getElementById("confThreshSlider");
  const confThreshVal = document.getElementById("confThreshVal");
  const imgszVal = document.getElementById("imgszVal");
  const confirmFramesVal = document.getElementById("confirmFramesVal");

  const imgszControl = document.getElementById("imgszControl");
  const confirmControl = document.getElementById("confirmControl");

  const videoStream = document.getElementById("videoStream");
  const videoPlaceholder = document.getElementById("videoPlaceholder");
  const sourceBadge = document.getElementById("sourceBadge");
  const videoFpsBadge = document.getElementById("videoFpsBadge");
  const procFpsBadge = document.getElementById("procFpsBadge");

  // Active Vehicle DOM Elements
  const activeTotal = document.getElementById("activeTotal");
  const activeCar = document.getElementById("activeCar");
  const activeMotorcycle = document.getElementById("activeMotorcycle");
  const activeBus = document.getElementById("activeBus");
  const activeTruck = document.getElementById("activeTruck");

  // Total Unique Vehicle DOM Elements
  const uniqueTotal = document.getElementById("uniqueTotal");
  const uniqueCar = document.getElementById("uniqueCar");
  const uniqueMotorcycle = document.getElementById("uniqueMotorcycle");
  const uniqueBus = document.getElementById("uniqueBus");
  const uniqueTruck = document.getElementById("uniqueTruck");

  // Analytics DOM Elements
  const trafficStateBadge = document.getElementById("trafficStateBadge");
  const densityBar = document.getElementById("densityBar");
  const densityVal = document.getElementById("densityVal");
  const queueLengthVal = document.getElementById("queueLengthVal");
  const avgWaitingTimeVal = document.getElementById("avgWaitingTimeVal");

  const eventLog = document.getElementById("eventLog");

  // State
  let ws = null;

  // Log Message Helper
  function logMessage(msg, type = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry log-entry-${type}`;
    const timeStr = new Date().toLocaleTimeString();
    entry.textContent = `[${timeStr}] ${msg}`;
    eventLog.prepend(entry);

    if (eventLog.children.length > 20) {
      eventLog.removeChild(eventLog.lastChild);
    }
  }

  // WebSocket Setup
  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/traffic`;

    logMessage(`Connecting WebSocket to ${wsUrl}...`, "info");
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      connectionStatus.className = "status-badge status-connected";
      statusText.textContent = "Live WS Connected";
      logMessage("WebSocket connection established.", "info");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleFrameData(data);
      } catch (err) {
        console.error("Error parsing WS frame data:", err);
      }
    };

    ws.onclose = () => {
      connectionStatus.className = "status-badge status-disconnected";
      statusText.textContent = "Disconnected";
      logMessage("WebSocket disconnected. Retrying in 3 seconds...", "error");
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      ws.close();
    };
  }

  // Process WebSocket Telemetry & Frame Payload
  function handleFrameData(payload) {
    if (payload.frame) {
      videoStream.src = payload.frame;
      videoStream.style.display = "block";
      videoPlaceholder.style.display = "none";
    }

    const telemetry = payload.telemetry;
    if (telemetry) {
      // 1. Update Active Vehicle Counters
      if (telemetry.active_vehicles) {
        updateActiveCounters(telemetry.active_vehicles);
      }

      // 2. Update Total Unique Vehicle Counters
      if (telemetry.unique_vehicles) {
        updateUniqueCounters(telemetry.unique_vehicles);
      }

      // 3. Update Traffic Analytics
      if (telemetry.traffic) {
        updateTrafficAnalytics(telemetry.traffic);
      }

      // 4. Update Performance FPS & Latency Badges
      if (telemetry.performance) {
        const procFps = telemetry.performance.processing_fps || 0;
        const videoFps = telemetry.performance.video_fps || 0;
        procFpsBadge.textContent = `Proc: ${procFps} FPS`;
        videoFpsBadge.textContent = `Video: ${videoFps} FPS`;
      }
    }
  }

  // Update Active Counters DOM
  function updateActiveCounters(active) {
    const car = active.car || 0;
    const motorcycle = active.motorcycle || 0;
    const bus = active.bus || 0;
    const truck = active.truck || 0;

    // Strict invariant: Active Total = Car + Motorcycle + Bus + Truck
    const total = car + motorcycle + bus + truck;

    activeCar.textContent = car;
    activeMotorcycle.textContent = motorcycle;
    activeBus.textContent = bus;
    activeTruck.textContent = truck;
    activeTotal.textContent = total;
  }

  // Update Total Unique Counters DOM
  function updateUniqueCounters(unique) {
    const car = unique.car || 0;
    const motorcycle = unique.motorcycle || 0;
    const bus = unique.bus || 0;
    const truck = unique.truck || 0;

    // Strict invariant: Unique Total = Car + Motorcycle + Bus + Truck
    const total = car + motorcycle + bus + truck;

    uniqueCar.textContent = car;
    uniqueMotorcycle.textContent = motorcycle;
    uniqueBus.textContent = bus;
    uniqueTruck.textContent = truck;
    uniqueTotal.textContent = total;
  }

  // Update Traffic Analytics DOM
  function updateTrafficAnalytics(traffic) {
    if (!traffic) return;

    const state = traffic.state || "LOW";
    trafficStateBadge.textContent = state;
    trafficStateBadge.className = "state-badge";

    if (state === "LOW") trafficStateBadge.classList.add("state-low");
    else if (state === "MODERATE") trafficStateBadge.classList.add("state-moderate");
    else if (state === "HIGH") trafficStateBadge.classList.add("state-high");
    else trafficStateBadge.classList.add("state-very-high");

    const densityPct = Math.round((traffic.density || 0) * 100);
    densityBar.style.width = `${densityPct}%`;
    densityVal.textContent = `${densityPct}%`;

    queueLengthVal.textContent = `${traffic.estimated_queue_length || 0} vehicles`;
    avgWaitingTimeVal.textContent = `${(traffic.estimated_average_waiting_time || 0).toFixed(1)} s`;
  }

  // EVENT LISTENERS

  // 1. Upload MP4 Video
  btnUploadVideo.addEventListener("click", () => {
    videoFileInput.click();
  });

  videoFileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    logMessage(`Uploading video '${file.name}'...`, "info");

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const res = await response.json();
      if (response.ok) {
        logMessage(`Video uploaded! Playing stream (Source FPS: ${res.fps}).`, "count");
        sourceBadge.textContent = `Video: ${file.name}`;
      } else {
        logMessage(`Upload failed: ${res.detail}`, "error");
        alert(`Error: ${res.detail}`);
      }
    } catch (err) {
      logMessage(`Upload exception: ${err.message}`, "error");
    }
  });

  // 2. Start Laptop Camera
  btnStartCamera.addEventListener("click", async () => {
    logMessage("Connecting to laptop camera...", "info");
    try {
      const response = await fetch("/api/camera/start", { method: "POST" });
      const res = await response.json();

      if (response.ok) {
        logMessage("Webcam stream started successfully.", "count");
        sourceBadge.textContent = "Laptop Webcam (Live)";
      } else {
        logMessage(`Camera error: ${res.detail}`, "error");
        alert(`Camera Error: ${res.detail}`);
      }
    } catch (err) {
      logMessage(`Camera start exception: ${err.message}`, "error");
    }
  });

  // 3. Stop Stream
  btnStopStream.addEventListener("click", async () => {
    try {
      await fetch("/api/camera/stop", { method: "POST" });
      videoStream.style.display = "none";
      videoPlaceholder.style.display = "flex";
      sourceBadge.textContent = "Stream Stopped";
      logMessage("Stream stopped.", "info");
    } catch (err) {
      console.error("Stop stream error:", err);
    }
  });

  // 4. Reset Counters
  btnResetCounters.addEventListener("click", async () => {
    try {
      await fetch("/api/reset", { method: "POST" });
      updateActiveCounters({ car: 0, motorcycle: 0, bus: 0, truck: 0 });
      updateUniqueCounters({ car: 0, motorcycle: 0, bus: 0, truck: 0 });
      logMessage("Counters reset to zero.", "info");
    } catch (err) {
      console.error("Reset error:", err);
    }
  });

  // 5. Confidence Threshold Slider
  confThreshSlider.addEventListener("input", (e) => {
    const val = (parseFloat(e.target.value) / 100.0).toFixed(2);
    confThreshVal.textContent = val;
  });

  confThreshSlider.addEventListener("change", async (e) => {
    const val = (parseFloat(e.target.value) / 100.0).toFixed(2);
    const formData = new FormData();
    formData.append("conf_threshold", val);

    try {
      await fetch("/api/config", { method: "POST", body: formData });
      logMessage(`Confidence threshold set to ${val}.`, "info");
    } catch (err) {
      console.error("Config update error:", err);
    }
  });

  // 6. Inference Resolution Segmented Control (640 / 960 / 1280)
  imgszControl.addEventListener("click", async (e) => {
    const btn = e.target.closest(".segment-btn");
    if (!btn) return;

    imgszControl.querySelectorAll(".segment-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    const val = parseInt(btn.dataset.val, 10);
    imgszVal.textContent = `${val}px`;

    const formData = new FormData();
    formData.append("imgsz", val);

    try {
      await fetch("/api/config", { method: "POST", body: formData });
      logMessage(`Inference resolution set to ${val}px.`, "info");
    } catch (err) {
      console.error("Config update error:", err);
    }
  });

  // 7. Track Confirmation Segmented Control (2f / 3f / 4f / 5f)
  confirmControl.addEventListener("click", async (e) => {
    const btn = e.target.closest(".segment-btn");
    if (!btn) return;

    confirmControl.querySelectorAll(".segment-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    const val = parseInt(btn.dataset.val, 10);
    confirmFramesVal.textContent = `${val} frames`;

    const formData = new FormData();
    formData.append("confirmation_frames", val);

    try {
      await fetch("/api/config", { method: "POST", body: formData });
      logMessage(`Track confirmation frames set to ${val}.`, "info");
    } catch (err) {
      console.error("Config update error:", err);
    }
  });

  // Initial WebSocket Connect
  connectWebSocket();
});
