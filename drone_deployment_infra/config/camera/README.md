# config/camera/

Per-drone camera calibration, e.g. `tello_4.yaml`: 960x720 pinhole K and
(k1, k2, p1, p2, k3), RMS, frames, coverage, date and source session.
`scripts/03_calibrate.py` writes these only when a calibration PASSes.
