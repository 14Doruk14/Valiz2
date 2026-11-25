# Owner-Following Suitcase Robot (Valiz 2)

A suitcase robot that follows its owner, built step by step from scratch.

| File | What it does |
|---|---|
| adim1_kamera.py | Open camera / video source |
| adim2_tespit.py | Person detection with YOLO11n |
| adim3_takip.py | IoU-based tracking with persistent IDs |
| adim4_sahip.py | Owner selection (automatic / mouse click) |
| adim5_konum.py | Horizontal error and distance estimation |
| adim6_hareket.py | Motion command (linear V / angular W) |
| adim7_reid.py | Re-identification: color histogram (7a) |
| adim7b_osnet.py | Re-identification: OSNet (7b, chosen) |
| adim8_olcum.py | Per-stage timing / FPS and video recording |

Not in the repo (too large): `yolo11n.pt` (downloaded automatically by ultralytics) and `modeller/osnet_x0_25_msmt17.onnx`.

## Roadmap
8 measurement + main program → 8.5 ROS 2 + Gazebo → 8.6 RL experiment (PPO) → 9 Raspberry Pi 5 → 10 ESP32 + motors → 11 real-world test
