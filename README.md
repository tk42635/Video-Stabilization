Video Stabilization Algorithm

## Dependency

- opencv-2 or opencv-3
- openmp for multi-threads

## Run

- Modify the COM of your camera or modify the path to a local video file
- Compile and Run

## Function Calls
- Call function VideoStab::Estimate() to return an affine matrix between two adjacent frames.
- Call function VideoStab::Compensate() to stabilize the latter frame between two adjacent frames

## Notes
This algorithm is based on KalmanFilter and GaussianFilter and contrived to run on embedded devices.
The two task, Estimate and Compensate, are divided into 2 independent running threads to meet the requirement of promoting computing efficiency.


