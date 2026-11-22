# Granular gas velocity extraction and analysis

[中文说明](README.zh-CN.md)

Supplementary code for *A Tabletop Granular-Gas Experiment Exploring Quasi-Maxwellian Velocity Statistics*. The workflow uses YOLOv8 and SORT to extract ball and container velocities, calibrate pixel dimensions, and generate velocity-distribution, KS, autocorrelation, and granular-temperature plots.

Run the commands below from this `code/` directory. Use the training section when preparing a new detector; with an existing `best.pt`, start with video processing.

## 1. Installation

Use Python 3.12 or newer. Install the dependencies listed in `requirements.txt`:

```sh
python -m pip install -r requirements.txt
```

Install Jupyter separately to use the notebook. For NVIDIA GPU execution, use a CUDA-enabled PyTorch installation. The extraction and measurement scripts select the device automatically unless `--device cpu` or `--device 0` is specified.

## 2. Directories and input preparation

| Location | Purpose |
| --- | --- |
| `1_TrainYolov8/` | Detector training, notebook, and evaluation |
| `1_TrainYolov8/dataset/` | YOLO dataset and `data.yaml` |
| `2_GetResults/video/` | Input experimental videos |
| `2_GetResults/best.pt` | Default detector weights for measurement and extraction |
| `2_GetResults/ball_size_measurements/` | Per-video ball-box measurements in Excel |
| `2_GetResults/processed_data/` | Calibration CSV, velocity CSVs, and diagnostic reports |
| `3_AnalysisResults/` | Analysis and plotting scripts |
| `3_AnalysisResults/figures/` | Exported PDF and PNG files |

Name videos `<rpm>r_<balls>.mp4`, for example `100r_10.mp4` for 10 balls at 100 rpm. Place them in `2_GetResults/video/`, or specify another directory with the commands below.

Generate ball-size measurements and velocities from the videos, summarize the measurements into `ball_sizes_means.csv`, and then run the analysis scripts. Analysis reads the calibration and velocity files from `2_GetResults/processed_data/`.

## 3. Ball-size calibration

Choose one measurement command, then run the summary command.

```sh
# Measure all MP4 videos in the default video directory
python 2_GetResults/measure_ball_sizes.py

# Alternatively, measure one video
python 2_GetResults/measure_ball_sizes.py --video 2_GetResults/video/100r_10.mp4

# Optional: discard sampled frames whose ball count does not match the filename
python 2_GetResults/measure_ball_sizes.py --require-ball-count

# Summarize measurements and replace the existing calibration table
python 2_GetResults/summarize_ball_sizes.py --overwrite
```

Measurement samples up to 100 equally spaced frames within the first 20000 video frames. Use `--max-frames N` to change the prefix, or `--max-frames 0` for the full video. `--video-dir PATH` selects another video directory; it is mutually exclusive with `--video`.

The optional `--require-ball-count` switch is disabled by default. When enabled, a sampled frame is retained only if its detected ball count matches the physical count in the filename (container detections are excluded). Mismatched frames are discarded without searching neighboring frames. The output reports sampled, valid and discarded frame counts and the number of saved ball boxes.

Detected box widths and heights are saved as `<condition>_ball_sizes.xlsx`. The summary script writes `filename,mean_width_px,mean_height_px` to `processed_data/ball_sizes_means.csv`. Analysis uses the row matching each condition. Summary options are `--input-dir`, `--output`, and `--overwrite`.

## 4. Velocity extraction

Choose single-video or batch processing as needed.

```sh
# One video, using default settings
python 2_GetResults/extract_velocities.py --video 2_GetResults/video/100r_10.mp4

# All videos in a directory, with size filtering and known-count enforcement
python 2_GetResults/batch_extract_velocities.py 2_GetResults/video --enforce-ball-count --ball-size-gate 0.85

# Serial processing of one video
python 2_GetResults/extract_velocities.py --video 2_GetResults/video/100r_10.mp4 --serial
```

The batch entry point processes `.mp4` and `.avi` files in the selected directory, one video at a time. Add `--serial` to the batch command for serial processing. The default limit is the first 20000 input frames per video; the number of velocity records depends on the detected objects and valid tracking intervals.

### Shared extraction and measurement options

| Option | Default | Meaning |
| --- | --- | --- |
| `--weights` | `2_GetResults/best.pt` | Detector weights |
| `--device` | auto | Computing device, e.g. `cpu` or `0` |
| `--max-frames` | `20000` | First N input frames; `0` uses the full video |
| `--conf` | `0.25` | YOLO detection confidence threshold |
| `--nms-iou` | `0.2` | YOLO NMS overlap threshold |
| `--tf32` | off | Enable TF32 convolution on compatible GPUs |
| `--output-dir` | By task | Extraction: `processed_data/`; measurement: `ball_size_measurements/` |

### Extraction-specific options

| Option | Default | Meaning |
| --- | --- | --- |
| `--batch-size` | `16` | Frames per inference call in parallel mode |
| `--prep-workers` | `4` | Frame-preparation threads; `0` prepares on the decoder thread |
| `--serial` | off | Process one frame at a time; ignores batch-size/prep-workers |
| `--sort-iou` | `0.3` | SORT association IoU threshold; use `--sort-iou 0.5` for stricter matching |
| `--expected-balls` | From filename | Override the physical ball count |
| `--ball-size-gate` | `0` | Lower size cutoff as a fraction of the running median; `0` disables it |
| `--enforce-ball-count` | off | Keep at most the expected number of ball boxes, ranked by confidence |

The size gate uses `sqrt(width*height)` and starts after collecting 100 sizes, retaining up to 2000 accepted sizes in its running history. Count enforcement follows size filtering. In a batch with different ball counts, omit `--expected-balls` so each video's count is read from its filename. A batch override applies to every video.

The `0.85` in `--ball-size-gate 0.85` is a ball-size ratio. The container tracker's template correlation threshold is also 0.85 and is enabled internally by default. Boolean switches such as `--serial`, `--tf32`, and `--enforce-ball-count` take no value; including the switch enables it.

## 5. Analysis and plotting

| Script | Output |
| --- | --- |
| `fig4_VelocityDistribution.py` | Lab-frame speed/component fits and enlarged speed-tail plot |
| `fig5_KSAnalysis.py` | KS distance versus effective sample size using Maxwell fits (`fig5_ks_fit_<condition>.*`) |
| `fig6_AutocorrelationHeatmap.py` | Velocity-autocorrelation decay-time heatmap |
| `fig7_TgHeatmap.py` | Lab-frame granular temperatures and KS distances |
| `fig4_VelocityDistribution_Relative.py` | Container-frame histograms and second-moment reference curves |
| `fig7_TgHeatmap_Relative.py` | Container-frame granular temperatures and KS distances |
| `fig4_VelocityDistribution_Moment.py` | Lab-frame second-moment reference curves |
| `fig5_KSAnalysis_Moment.py` | KS convergence using second-moment references (`fig5_ks_moment_<condition>.*`) |

### Main-text figures

```sh
# Figure 4: laboratory-frame distributions for the two representative conditions
python 3_AnalysisResults/fig4_VelocityDistribution.py --condition 100r_10
python 3_AnalysisResults/fig4_VelocityDistribution.py --condition 120r_10

# Figure 5: KS convergence
python 3_AnalysisResults/fig5_KSAnalysis.py

# Figure 6: velocity autocorrelation
python 3_AnalysisResults/fig6_AutocorrelationHeatmap.py

# Figure 7: laboratory-frame speed distributions and granular temperatures
python 3_AnalysisResults/fig7_TgHeatmap.py
```

### Optional comparisons

```sh
# Container-frame distributions and granular temperatures
python 3_AnalysisResults/fig4_VelocityDistribution_Relative.py --condition 100r_10
python 3_AnalysisResults/fig7_TgHeatmap_Relative.py

# Laboratory-frame second-moment comparisons
python 3_AnalysisResults/fig4_VelocityDistribution_Moment.py --condition 100r_10
python 3_AnalysisResults/fig5_KSAnalysis_Moment.py
```

### Usage notes

All plotting scripts export PDF and PNG files and accept `--pdf-dir`. For example, `--pdf-dir figures` writes into `3_AnalysisResults/figures/`. Relative plot-output paths are resolved from the analysis directory; other custom paths are resolved from the working directory.

For Figures 4 and 7, laboratory-frame analyses use all ball samples with finite velocity components, without requiring a container velocity in the same frame. Container-frame analyses use only same-frame pairs with valid ball and container velocities, subtracting the container velocity componentwise. The two frames may therefore use different sample counts.

All three Figure 4 scripts default to `100r_10`. Figure 5 uses `120r_10`; change `target_rpm` and `target_balls` in its `main()` for another condition. Grid conditions are set by `rpm_list` and `ball_list` in the corresponding scripts. The default Figure 6/7 rpm list is 40, 60, 80, 100, 120, 140, 180, 200. Figure 6 uses 2–10 balls; Figure 7 uses 2, 4, 6, 8, 10 balls. Missing inputs appear as `N/A`; unavailable fits or estimates appear as `Fail`.

The Figure 4 laboratory-frame speed panel shows 0–0.3 m/s to keep the main peak readable. The tail plots show 0.10–0.25 m/s for `100r_10` and 0.12–0.30 m/s for `120r_10`; other conditions default to 0.10–0.30 m/s. The `_Moment.py` script uses the corresponding windows. These axis limits do not change histogram normalization or the full samples used for fitting, KS distances and second moments.

Figure 7 panels share a display range covering at least 99.9% of each condition's samples within each rpm column. This display window changes axis limits only: histograms are normalized over all valid samples, and all valid samples contribute to fits, KS distances and second moments.

## 6. Model training and evaluation (optional)

Classes are `0 = ball` and `1 = container`. Under `1_TrainYolov8/`, place images and same-stem YOLO labels in `dataset/train/images`, `dataset/train/labels`, `dataset/val/images`, and `dataset/val/labels`. Each label row contains normalized `class cx cy width height`. Configure the split paths and class names in `1_TrainYolov8/dataset/data.yaml`; a test split may also be provided.

```sh
# Train with the script's default settings
python 1_TrainYolov8/yolo_train.py --epochs 100 --batch 32 --imgsz 640 --workers 4 --seed 0

# Evaluate the detector used for video processing
python 1_TrainYolov8/yolo_evaluate.py --weights 2_GetResults/best.pt --split val

# Optional test-set evaluation
python 1_TrainYolov8/yolo_evaluate.py --weights 2_GetResults/best.pt --split test
```

Training also accepts `--model`, `--optimizer`, and `--lr0`. The default `--optimizer auto` selects the optimizer and learning rate automatically. To specify them manually, use, for example, `--optimizer AdamW --lr0 0.001`.

Both training entry points delete the previous `1_TrainYolov8/runs/train/` directory when training starts, without a backup, and save the new results to that same directory. The best weights are `1_TrainYolov8/runs/train/weights/best.pt`; use this path as `--weights` for both size measurement and velocity extraction. Evaluation reports matched-box center/size errors and IoU. The notebook `yolo_train.ipynb` provides interactive training and detector demonstrations; open it from `1_TrainYolov8/` and adjust its input paths as needed. Only the training cell clears old results; evaluation and demonstration cells do not.

## 7. Outputs and calculation settings

| File | Contents |
| --- | --- |
| `ball_size_measurements/<condition>_ball_sizes.xlsx` | Detected ball-box widths and heights in pixels |
| `processed_data/ball_sizes_means.csv` | Condition-specific mean box dimensions for calibration |
| `processed_data/vx_vy_data_<condition>.csv` | `frame,timestamp,obj_id,vx,vy`; timestamps in seconds, velocities in px/s |
| `processed_data/frame_counts_<condition>.csv` | Per-frame detection/filtering/tracking/velocity counts and container validity |
| `processed_data/frame_counts_<condition>.json` | Count summaries, filter settings, and detailed removal events |

Here the two output directories are inside `2_GetResults/`. In velocity CSVs, `obj_id` is a SORT track ID, or `container` for container velocities. Re-running a video with the same stem replaces its corresponding outputs.

In diagnostic JSON, `ball_filtering` summarizes removed-box totals and affected-frame totals. `ball_filter_events` contains each affected frame's boxes, sizes, confidences, decisions, and removal reasons. It also records the minimum retained confidence, all removed confidence scores, and the count-selection boundary gap. An empty event array is `[]`. Counts represent detection-box occurrences across frames. The diagnostic CSV includes raw/post-size-gate/final counts and the size threshold actually applied in each frame.

### Motion and physical units

The experimental videos were recorded in portrait orientation, so both size measurement and velocity extraction rotate decoded frames 90 degrees counterclockwise to restore an upright view. In the rotated images, x points right and y points down. Ball velocities use SORT-box-center displacements between consecutive video frames and the actual timestamps. Gaps break the velocity calculation. Container positions use YOLO initialization and dual-corner rim-mask matching, with a template correlation threshold of 0.85. Unreliable positions are omitted; container velocities use linear least-squares fits over four consecutive valid positions, aligned with the ball-velocity interval. Missing velocities are not interpolated.

Analysis uses a sphere diameter of 0.014 m and mass of 0.0036 kg. Pixel-to-metre conversion is condition-specific:

```text
d_px = (mean_width_px + mean_height_px) / 2
scale = 0.014 / d_px
v_m_per_s = v_px_per_s * scale
```

Update the physical constants in the analysis scripts when using different spheres. Lab-frame temperatures use two-dimensional Maxwell speed fits, with zero-mean Gaussian fits for velocity components; container-frame temperatures use second moments, with Maxwell/Gaussian curves as references. The `_Moment.py` scripts provide the corresponding lab-frame moment comparisons. Second moments use the measured velocities without subtracting their mean.

## License

See [LICENSE](LICENSE), the bundled SORT license in [LICENSES/GPL-3.0.txt](LICENSES/GPL-3.0.txt), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
