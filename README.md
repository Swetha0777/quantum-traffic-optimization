# Quantum-Enhanced Adaptive Urban Traffic Optimization

A modular hackathon project for real-time traffic signal optimization using Quantum Algorithms (QUBO & QAOA) and dynamic emergency green corridor routing.

---

## 👥 Team Module Ownership

| Team Member | Module Path | Function / Responsibility |
| :--- | :--- | :--- |
| **Member 1** | `simulation/traffic_simulation.py` | Traffic generation & flow simulation for different urban scenarios |
| **Member 2** | `optimization/qubo.py` & `optimization/qaoa.py` | Formulates QUBO matrices and runs QAOA signal timing solver |
| **Member 3** | `emergency/emergency_corridor.py` | Dynamic emergency green corridor routing |
| **Member 4** | `metrics/performance.py` | Computes performance metrics (delay, throughput, emissions) |

---

## 📁 Directory Structure

```text
quantum_traffic/
│
├── app.py                      # Main Streamlit dashboard application
│
├── data/
│   └── traffic_data.csv        # Dataset storage for traffic data
│
├── simulation/                 # Module 1: Traffic Simulation (Member 1)
│   ├── __init__.py
│   └── traffic_simulation.py
│
├── optimization/               # Module 2: Quantum Optimization (Member 2)
│   ├── __init__.py
│   ├── qubo.py
│   └── qaoa.py
│
├── emergency/                  # Module 3: Emergency Corridor (Member 3)
│   ├── __init__.py
│   └── emergency_corridor.py
│
├── metrics/                    # Module 4: Metrics & Performance (Member 4)
│   ├── __init__.py
│   └── performance.py
│
├── utils/                      # Shared constants & helpers
│   ├── __init__.py
│   └── constants.py
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 How to Run the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
Navigate into the `quantum_traffic` directory and run:
```bash
cd quantum_traffic
streamlit run app.py
```
