# Legacy

Superseded work, kept for reference. Nothing here is needed to run the current system.

- `v1_image_heatmaps/`: the first model. It predicted waypoint heatmaps in image pixels. It reached
  about 3 px error on an unseen track but failed on bends, because waypoints that leave the camera
  image could not be supervised. Replaced by the top-down model in `scripts/trace_bev.py`.
- `carla/`: recording and labelling scripts written for CARLA before the project moved to Gazebo.
  They were never run against CARLA.
