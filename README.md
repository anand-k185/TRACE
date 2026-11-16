# TRACE

**Camera-only path prediction for a small autonomous car.** A vision transformer looks at one front-camera frame and predicts where the car should drive next. A simple controller follows that path. No map, lidar or odometry is used while driving.

This repository holds the full proof of concept: data recording in simulation, labelling, training, inspection, closed-loop driving and scoring.

> Status: research prototype, simulation only. See [Limitations](#limitations).

---

## How it works

```
 camera frame ──► ViT encoder ──► top-down decoder ──► 6 waypoints (meters) ──► pure pursuit ──► steering
   320×240        (pretrained)     (cross-attention)     1.0–2.5 m ahead
```

**Model**

| Part | Details |
|---|---|
| Encoder | ViT-Small/16 from `timm` (`vit_small_patch16_224.augreg_in21k_ft_in1k`), fine-tuned. 196 patch tokens per image. |
| Geometry prior | Each cell of a floor grid around the car starts from the image feature where the camera sees that cell (inverse perspective mapping, from the camera's own calibration). |
| Decoder | 640 floor queries, 3 transformer layers of self-attention and cross-attention to all image tokens. This fills in parts of a bend the camera cannot see. |
| Output | One heatmap per waypoint on a top-down grid: 0–4 m ahead, ±3.2 m sideways, 8 cm cells. The decoded peak is the waypoint in meters. |

**Training data.** An autopilot that has the track map drives each training track while the camera is recorded. It deliberately weaves up to 25 cm off-center, so the model sees what recovering looks like. Each frame is labelled with the track centerline ahead of the car. The map is used for labels and for grading only; it is never a model input.

**Why top-down heatmaps.** The first version predicted heatmaps in image pixels. On tight bends most of the path leaves the camera's field of view, so those waypoints could not be supervised and the model failed on curves. Predicting on the floor grid supervises every waypoint on every frame. The image-heatmap version is kept in `legacy/` for reference.

---

## Results so far

| Test | Result |
|---|---|
| Closed loop on `lab_15`, a track never used for training (15 Nov 2026) | 67.8 m driven (0.75 lap; the recording was stopped early), 0 duct contacts, never stuck, 0.11 m average distance from the centerline, 2.4 cm closest approach |
| Training set | 15 tracks (`lab_00`–`lab_14`), about 30,300 labelled frames |
| Held-out tracks | `lab_15` (used during development), `lab_16`–`lab_20` (untouched) |

A seen-versus-unseen comparison across 11 tracks is set up in `scripts/test_all.py` and `scripts/compare_results.py`. Add its table and `robustness.png` here once the run is complete.

---

## Repository layout

```
scripts/      Everything that runs next to the simulator (WSL / Ubuntu)
  record_all.py        record many tracks unattended (sim + autopilot + recorder)
  autodrive.py         map-based autopilot used as the teacher; --weave for off-center driving
  gz_collect.py        saves camera frames with the car's pose
  gz_make_labels.py    turns recordings into waypoint labels
  model_drive.py       runs the trained model on live camera frames and steers the car
  eval_drive.py        scores one recorded drive against the map
  test_all.py          drives and scores the model on several tracks unattended
  compare_results.py   seen-vs-unseen table and figure from test_all.py results
  trace_bev.py         model, loss and decoding (shared by training and driving)
notebooks/    Google Colab
  TRACE_train_bev.ipynb     training
  TRACE_inspect_bev.ipynb   heatmaps and paths vs labels and centerline, error tables
  camera_heatmap_cell.py    optional cell: heatmaps projected into the camera view
  trace_ref_bev.py          plotting and metric helpers (embedded in the inspect notebook)
cluster/      Command-line training for a SLURM GPU cluster (UNTESTED, see below)
legacy/       Superseded work kept for reference
  v1_image_heatmaps/   first model: heatmaps in image pixels (fails on bends)
  carla/               early CARLA recording scripts (never run)
```

The scripts in `scripts/` import each other, so keep them in one folder and run them from there.

---

## Requirements

| Item | Details |
|---|---|
| OS | Ubuntu 22.04 (tested under WSL2 on Windows, Intel integrated graphics) |
| ROS | ROS 2 Humble |
| Simulator | [jfasate/f1tenth_gym_gazebo](https://github.com/jfasate/f1tenth_gym_gazebo) on Gazebo Classic 11 |
| Python (WSL) | system Python 3.10 with `numpy`, `pillow`, `scipy`, `pyyaml`, `matplotlib`, plus CPU `torch` and `timm` for driving |
| Training | Google Colab with a T4 GPU, or any CUDA GPU |

---

## Quick start: drive with a trained checkpoint

You need a checkpoint, `trace_vit_bev.pt`. It is too large for the repository; get it from the project's shared drive or the GitHub release.

### 1. Install the simulator

```bash
sudo apt update
sudo apt install -y ros-humble-gazebo-ros-pkgs ros-humble-rqt-image-view \
  python3-colcon-common-extensions python3-rosdep python3-numpy python3-pil \
  python3-scipy python3-yaml python3-matplotlib git
sudo rosdep init 2>/dev/null; rosdep update

mkdir -p ~/sim_ws/src && cd ~/sim_ws/src
git clone https://github.com/jfasate/f1tenth_gym_gazebo.git
cd ~/sim_ws
rosdep install --from-paths src --ignore-src -y --skip-keys ament_python
```

On slow machines Gazebo takes longer to start than the 30 seconds the simulator's spawner waits, and the car never appears. Raise the limit before building:

```bash
sed -i "s/'-Y', yaw\],/'-Y', yaw, '-timeout', '120'],/" \
  ~/sim_ws/src/f1tenth_gym_gazebo/launch/sim_launch.py
colcon build
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/sim_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Install this repository and PyTorch

```bash
git clone <this repository> ~/TRACE
cd ~/TRACE/scripts
cp /path/to/trace_vit_bev.pt .

pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install --user --no-deps timm
pip3 install --user huggingface_hub safetensors
```

`--no-deps` keeps pip from replacing the numpy that ROS depends on. If `colcon build` later fails with `canonicalize_version() got an unexpected keyword argument`, pip installed a newer `setuptools` in your user folder; remove it with `pip3 uninstall -y setuptools`.

### 3. Run

Use three terminals, all in `~/TRACE/scripts`.

```bash
# Terminal 1: simulator on a track the model has never seen. Wait for "Gazebo write path OK".
ros2 launch f1tenth_gym_gazebo sim_launch.py map_name:=lab_19 enable_lidar3d:=false \
  gui:=false rviz:=false camera_image_width:=320 camera_image_height:=240

# Terminal 2: record the drive so it can be scored
python3 gz_collect.py --map lab_19 --out test_drives/ --hz 5

# Terminal 3: the model drives
python3 model_drive.py --ckpt trace_vit_bev.pt --speed 0.6 --duration 300
```

Watch the predicted path on the camera image:

```bash
ros2 run rqt_image_view rqt_image_view /trace/overlay
```

When the model stops, press Ctrl-C in the recorder, then the simulator, and score the drive:

```bash
python3 eval_drive.py --episode $(ls -d test_drives/lab_19/ep_* | tail -1) --map lab_19
```

This prints laps, duct contacts, closest approach and distance from the centerline, and saves `drive_eval.png` in the episode folder.

Notes:

- Always pass `map_name:=lab_XX`. Only `lab_00`–`lab_20` launch as shipped; the simulator's default map and its other named tracks are missing map files.
- Use `gui:=true` to see the Gazebo window. It costs frame rate on weak graphics.
- The camera size must be 320×240, the same as in the training recordings.

`model_drive.py` options:

| Option | Default | Use it when |
|---|---|---|
| `--speed` | 0.8 | Lower to 0.4 if the car touches ducts on hairpins |
| `--lookahead` | 1.4 | Raise if it weaves on straights; lower if it cuts corners |
| `--steer-gain` | 1.0 | Lower to about 0.85 if it turns too hard into bends |
| `--smooth` | 0.0 | Raise to about 0.3 if the steering is jerky |
| `--threads` | 4 | Adjust if the log shows more than about 300 ms per frame |

---

## Test robustness on many tracks

```bash
cd ~/TRACE/scripts
python3 test_all.py --ckpt trace_vit_bev.pt --duration 240 --speed 0.6 \
  --maps lab_03 lab_08 lab_09 lab_11 lab_14 lab_15 lab_16 lab_17 lab_18 lab_19 lab_20
python3 compare_results.py
```

`test_all.py` starts the simulator, drives and scores each track in both directions, unattended (22 drives, 2.5 to 3 hours). `compare_results.py` groups the drives into **Trained on** (`lab_00`–`lab_14`), **Development** (`lab_15`) and **Brand new** (`lab_16`–`lab_20`) and writes:

| File | Content |
|---|---|
| `robustness.png` | One panel per measure, one mark per drive, grouped |
| `robustness_table.csv` | One row per group |
| `robustness_drives.csv` | One row per drive |

---

## Full pipeline: record, label, train

Only needed to retrain or add data.

### 1. Record

```bash
cd ~/TRACE/scripts
python3 record_all.py --maps lab_03                  # trial: 2 runs, about 10 minutes
python3 record_all.py                                # 15 training tracks, both directions, 2-3 hours
python3 record_all.py --maps lab_15 --weave 0        # held-out test track, no weaving
```

For each track and direction the script starts the simulator, waits for the car (spawning it itself if the simulator's spawner gives up), runs the recorder and two laps of the weaving autopilot, then shuts down. Recordings go to `data/lab_XX/ep_*/`.

### 2. Label

```bash
for d in data/lab_*; do
  python3 gz_make_labels.py --data $d --labels centerline --start 1.0 --spacing 0.3 --viz 10
done
tar czf trace_data.tar.gz --exclude='*/viz' data/
```

Each label is 6 points on the map centerline, 1.0 to 2.5 m ahead of the car, in the car's frame. Check a few images in `data/lab_XX/viz/`: the green line should run between the ducts and lead back to the middle when the car is off-center.

### 3. Train (Colab)

1. Upload `trace_data.tar.gz` to the top level of Google Drive.
2. Open `notebooks/TRACE_train_bev.ipynb` in Colab, set **Runtime → Change runtime type → T4 GPU**, and run all cells.
3. The best checkpoint is saved to `My Drive/trace_runs/trace_vit_bev.pt`.

The notebook trains on every `lab_*` folder except `lab_15`, for 12 epochs with `curve_boost=1.0`, and reports errors in meters for curves and straights on the seen tracks and on `lab_15`.

### 4. Inspect (Colab)

Run `notebooks/TRACE_inspect_bev.ipynb`. For each frame it shows the predicted heatmaps, the predicted path against the label and the map centerline, and the camera view, plus error tables. Paste `camera_heatmap_cell.py` into a new cell to see the heatmaps projected into the camera image.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `SetEntityState: entity [f1tenth] does not exist` repeats forever | The car never spawned. Apply the `-timeout` patch above, or spawn by hand with `ros2 run gazebo_ros spawn_entity.py -entity f1tenth -topic ego_robot_description -x X -y Y -z 0.0 -Y YAW -timeout 120`, using the pose from `maps/generated_safe_maps_spawns.yaml` in the simulator repository. The error is normal for the first seconds of every launch. |
| `Refusing to launch while old sim/RViz nodes are still running` | `pkill -f gzserver; pkill -f gzclient; pkill -f robot_state_publisher; pkill -f dynamics_plant; pkill -f map_publisher` |
| `rosdep` cannot resolve `ament_python` | Add `--skip-keys ament_python` |
| Gazebo crashes or the camera is black under WSL | `export LIBGL_ALWAYS_SOFTWARE=1` before launching |
| Camera runs at about 5 Hz | Expected on integrated graphics. Keep 320×240 and `gui:=false`. |
| `model_drive.py: unrecognized arguments` | An old copy of `model_drive.py` is in use; take the one from this repository |
| Colab prints `CUDA: False` | The runtime has no GPU selected |
| Colab trains on old data | Runtime → Disconnect and delete runtime, then run all |

---

## Limitations

- **Simulation only.** Nothing has run on a real car.
- **One visual environment.** Every track has the same ducts, floor and lighting. The tests show robustness to new layouts, not to new surroundings.
- **Labels need a map.** Each training track needs an occupancy map to trace its centerline.
- **Flat floor and fixed camera.** The geometry prior assumes both.
- **Low frame rate.** The test machine delivers about 5 camera frames per second, and reaction delay is not compensated.
- **No obstacles, no commands.** The model follows the corridor it sees and cannot yet avoid a block or choose a branch.
- **`cluster/` is untested.** `train_bev.py` and the SLURM job file were written for Purdue's Gilbreth cluster but have not been run. Set the account and partition in the job file before trying them.

## Roadmap

1. Finish the seen-versus-unseen robustness comparison.
2. Static obstacle avoidance: blocks on the track, an avoiding teacher, the same model.
3. Command conditioning (follow / take left / take right) on the simulator's fork and junction tracks.
4. A vision-language model (for example Qwen2.5-VL) at low rate to recognize exits and issue that command.
5. Recovery training from the model's own drives (DAgger) and delay compensation.
6. More visual variety, then a real F1TENTH car.

## Acknowledgements

- Simulator: [jfasate/f1tenth_gym_gazebo](https://github.com/jfasate/f1tenth_gym_gazebo)
- Encoder architecture: Dosovitskiy et al., *An Image is Worth 16x16 Words* ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929)); pretrained weights via [timm](https://github.com/huggingface/pytorch-image-models)
