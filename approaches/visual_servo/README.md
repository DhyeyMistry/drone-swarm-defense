# Visual servo defense: camera-only detection, tracking and interception

> 🚧 **More coming soon:** the detector training pipeline, the tracker and forecaster, the "Drone hunter" interface, the plots and the demo video will be added to this folder. The numbers below are the results presented to the judges.

![Drone hunter: live detection, 3-D view and targets](../../docs/media/presentation/visual_servo_drone_hunter.jpg)

*"Drone hunter":
- **left:** the defender's camera, with the detected drone and the forecast rings;
- **right:** a 3-D view of our drone, the target and its predicted paths;
- **bottom:** target list and detector controls.

This is a still from the demo video shown in the presentation.*

## Approach

- **Three Tello defenders**, each with its own camera vision, plus team coordination.
- **No labelled data needed:**
  - we auto-labelled our own flight footage with **YOLOE**;
  - recovered far-away drones using zoomed crops;
  - added **4,000 synthetic images**.
- **YOLO26 segmentation**, trained in about **35 minutes on an H100**. On unseen footage it detects **100% of nearby drones** and **91% at mid-range**.
- **Depth-aware tracking** plus a **Social-GAN-style forecaster** that predicts **20 possible paths per attacker**. The defender aims for the earliest point it can reach that most of those paths pass through.
- **Team play:** the fastest-reaching defender intercepts while the other two cover the base. It works from published positions **or from the cameras alone**.

## Results

- **Detection:** 100% of nearby drones and 91% at mid-range, on unseen footage.
- **Interception:** **3/3 intercepts in simulation.**

## To be added

- [ ] Auto-labelling pipeline (YOLOE, zoomed crops) and the synthetic-image generator
- [ ] YOLO26 segmentation training configs, weights and detection metrics
- [ ] Depth-aware tracker and the Social-GAN-style forecaster
- [ ] "Drone hunter" interface and the interception controller
- [ ] Plots, the demo video and the simulation results
