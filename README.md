# webcam-segmetation

- [x] Read from from v4l device
- [x] Run segmentation inference and obtain mask
- [x] Crop frames using mask provided by model 
- [x] Cleanup
  - [x] Fix colorspace
  - [x] Optimize if needed
- [x] Write cropped frames to loopback v4l device
- [x] Integrate with obs
- [ ] Fix glitching hands
- [ ] Send empty frames when no detections are found
- [ ] Long-running reliability
