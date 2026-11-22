# 颗粒气体速度提取与统计分析

[English](README.md)

本程序是论文 *A Tabletop Granular-Gas Experiment Exploring Quasi-Maxwellian Velocity Statistics* 的配套代码，使用 YOLOv8 和 SORT 提取球与容器的速度，完成像素尺寸标定，并绘制速度分布、KS 收敛、自相关和颗粒温度图。

以下命令均从本 README 所在的 `code/` 目录运行。需要建立新检测模型时使用训练部分；使用已有 `best.pt` 时，可直接进行视频处理。

## 1. 安装

使用 Python 3.12 或更高版本，安装 `requirements.txt` 中列出的依赖：

```sh
python -m pip install -r requirements.txt
```

使用 notebook 时另行安装 Jupyter。使用 NVIDIA GPU 时，安装支持 CUDA 的 PyTorch。提取与测量脚本默认自动选择设备，也可通过 `--device cpu` 或 `--device 0` 指定。

## 2. 目录与输入准备

| 位置 | 用途 |
| --- | --- |
| `1_TrainYolov8/` | 检测模型训练、notebook 和评估 |
| `1_TrainYolov8/dataset/` | YOLO 数据集和 `data.yaml` |
| `2_GetResults/video/` | 输入实验视频 |
| `2_GetResults/best.pt` | 尺寸测量和速度提取的默认模型权重 |
| `2_GetResults/ball_size_measurements/` | 各视频的球框尺寸测量 Excel |
| `2_GetResults/processed_data/` | 标定表、速度 CSV 和诊断报告 |
| `3_AnalysisResults/` | 统计分析与绘图脚本 |
| `3_AnalysisResults/figures/` | 导出的 PDF 和 PNG |

视频使用 `<转速>r_<球数>.mp4` 命名，例如 `100r_10.mp4` 表示 100 rpm、10 个球。将视频放入 `2_GetResults/video/`，或通过下述命令指定其他目录。

处理流程为：从视频测量球框尺寸、提取速度，将尺寸结果汇总为 `ball_sizes_means.csv`，再运行分析脚本。分析脚本直接读取 `2_GetResults/processed_data/` 中的标定表和速度 CSV。

## 3. 球尺寸标定

选择以下一种测量方式，再运行汇总命令。

```sh
# 测量默认 video/ 目录中的全部 MP4 视频
python 2_GetResults/measure_ball_sizes.py

# 或者只测量一个视频
python 2_GetResults/measure_ball_sizes.py --video 2_GetResults/video/100r_10.mp4

# 可选：球数与文件名不符时，丢弃整个采样帧
python 2_GetResults/measure_ball_sizes.py --require-ball-count

# 汇总测量结果，覆盖已有标定表
python 2_GetResults/summarize_ball_sizes.py --overwrite
```

默认在视频前 20000 帧内等间隔抽取最多 100 帧。通过 `--max-frames N` 修改范围，`--max-frames 0` 使用整个视频。`--video-dir 路径` 可指定其他视频目录，与 `--video` 互斥。

`--require-ball-count` 是默认关闭的可选开关。开启后，只有检测球数与文件名中的实际球数一致时才保留该采样帧，容器不计入球数；不符则整帧丢弃，不搜索邻帧补足。输出报告采样帧数、有效帧数、丢弃帧数和保存的球框总数。

检测框的宽高保存为 `<工况>_ball_sizes.xlsx`。汇总脚本将 `filename,mean_width_px,mean_height_px` 写入 `processed_data/ball_sizes_means.csv`，分析时按工况读取对应行。汇总参数为 `--input-dir`、`--output` 和 `--overwrite`。

## 4. 速度提取

按需选择单视频或批量处理命令。

```sh
# 按默认设置提取一个视频
python 2_GetResults/extract_velocities.py --video 2_GetResults/video/100r_10.mp4

# 批量处理一个目录，并启用尺寸筛选和实际球数约束
python 2_GetResults/batch_extract_velocities.py 2_GetResults/video --enforce-ball-count --ball-size-gate 0.85

# 单视频逐帧串行处理
python 2_GetResults/extract_velocities.py --video 2_GetResults/video/100r_10.mp4 --serial
```

批量入口依次处理指定目录下的 `.mp4` 和 `.avi` 视频；添加 `--serial` 可改为串行处理。每个视频默认处理前 20000 个输入帧，最终速度记录数取决于检测对象及有效跟踪区间。

### 提取与尺寸测量共用参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--weights` | `2_GetResults/best.pt` | 检测模型权重 |
| `--device` | 自动选择 | 计算设备，例如 `cpu` 或 `0` |
| `--max-frames` | `20000` | 使用前 N 个输入帧；`0` 使用整个视频 |
| `--conf` | `0.25` | YOLO 检测置信度阈值 |
| `--nms-iou` | `0.2` | YOLO NMS 重叠阈值 |
| `--tf32` | 关闭 | 在兼容 GPU 上启用 TF32 卷积运算 |
| `--output-dir` | 按任务区分 | 提取为 `processed_data/`；尺寸测量为 `ball_size_measurements/` |

### 速度提取专用参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--batch-size` | `16` | 并行模式每次推理的帧数 |
| `--prep-workers` | `4` | 预处理线程数；`0` 表示在解码线程中预处理 |
| `--serial` | 关闭 | 逐帧处理，忽略 batch-size/prep-workers |
| `--sort-iou` | `0.3` | SORT 轨迹关联 IoU 阈值；使用 `--sort-iou 0.5` 可收紧匹配要求 |
| `--expected-balls` | 从文件名读取 | 手动指定实际球数 |
| `--ball-size-gate` | `0` | 尺寸下限占滚动中位数的比例；`0` 关闭 |
| `--enforce-ball-count` | 关闭 | 按置信度保留不超过实际球数的检测框 |

尺寸筛选使用 `sqrt(宽×高)`，收集 100 个尺寸后开始应用阈值，滚动历史最多保留 2000 个已接受尺寸。球数约束在尺寸筛选之后执行。批量处理不同球数时，省略 `--expected-balls`，让程序分别从每个视频的文件名读取球数；手动指定时，该球数用于批次内所有视频。

`--ball-size-gate 0.85` 中的 0.85 是球框尺寸比例。容器模板匹配另有内部默认启用的 0.85 相关分数阈值。`--serial`、`--tf32`、`--enforce-ball-count` 都是布尔开关，直接添加参数即可开启，后面无需填写数值。

## 5. 分析与绘图

| 脚本 | 输出内容 |
| --- | --- |
| `fig4_VelocityDistribution.py` | 实验室系速率和分量拟合，以及速率尾部放大图 |
| `fig5_KSAnalysis.py` | 使用 Maxwell 拟合的 KS 距离与有效样本量关系，输出 `fig5_ks_fit_<工况>.*` |
| `fig6_AutocorrelationHeatmap.py` | 速度自相关衰减时间热图 |
| `fig7_TgHeatmap.py` | 实验室系颗粒温度和 KS 距离 |
| `fig4_VelocityDistribution_Relative.py` | 容器系直方图和二阶矩参考曲线 |
| `fig7_TgHeatmap_Relative.py` | 容器系颗粒温度和 KS 距离 |
| `fig4_VelocityDistribution_Moment.py` | 实验室系二阶矩参考曲线 |
| `fig5_KSAnalysis_Moment.py` | 使用二阶矩参考的 KS 收敛分析，输出 `fig5_ks_moment_<工况>.*` |

### 主文绘图

```sh
# 图 4：两个代表工况的实验室系速度分布
python 3_AnalysisResults/fig4_VelocityDistribution.py --condition 100r_10
python 3_AnalysisResults/fig4_VelocityDistribution.py --condition 120r_10

# 图 5：KS 收敛
python 3_AnalysisResults/fig5_KSAnalysis.py

# 图 6：速度自相关
python 3_AnalysisResults/fig6_AutocorrelationHeatmap.py

# 图 7：实验室系速率分布与颗粒温度
python 3_AnalysisResults/fig7_TgHeatmap.py
```

### 可选补充分析

```sh
# 容器系速度分布与颗粒温度
python 3_AnalysisResults/fig4_VelocityDistribution_Relative.py --condition 100r_10
python 3_AnalysisResults/fig7_TgHeatmap_Relative.py

# 实验室系二阶矩对照
python 3_AnalysisResults/fig4_VelocityDistribution_Moment.py --condition 100r_10
python 3_AnalysisResults/fig5_KSAnalysis_Moment.py
```

### 使用说明

所有绘图脚本均导出 PDF 和 PNG，并支持 `--pdf-dir`。例如，`--pdf-dir figures` 输出至 `3_AnalysisResults/figures/`。绘图输出的相对路径以分析目录为基准；其他自定义路径以运行命令时的工作目录为基准。

图 4 和图 7 的实验室系使用球速度两个分量均为有限值的全部样本，不要求同帧存在容器速度；容器系仅使用同帧球速度与容器速度均有效的配对样本，按分量相减得到相对速度。因此，两种参考系的样本数可能不同。

图 4 的三个脚本均默认使用工况 `100r_10`。图 5 使用 `120r_10`，可修改其 `main()` 中的 `target_rpm` 和 `target_balls`。网格图工况由相应脚本中的 `rpm_list` 和 `ball_list` 设置：图 6/7 默认转速为 40、60、80、100、120、140、180、200 rpm；图 6 使用 2–10 个球，图 7 使用 2、4、6、8、10 个球。`N/A` 表示缺少该工况输入，`Fail` 表示当前数据未能完成拟合或统计。

图 4 实验室系速率主图显示 0–0.3 m/s，以便看清主峰；尾图中，`100r_10` 显示 0.10–0.25 m/s，`120r_10` 显示 0.12–0.30 m/s，其他工况默认显示 0.10–0.30 m/s。`_Moment.py` 脚本使用对应的尾图窗口。这些坐标范围不影响直方图归一化，以及拟合、KS 距离和二阶矩所使用的完整样本。

图 7 在每个转速列内共享横轴，显示范围覆盖该列每个工况至少 99.9% 的样本。这里只调整坐标轴：直方图按全部有效样本归一化，拟合、KS 距离和二阶矩仍使用全部有效样本。

## 6. 模型训练与评估（可选）

类别为 `0 = ball`（球）和 `1 = container`（容器）。在 `1_TrainYolov8/` 下，将图像及同名 YOLO 标注分别放入 `dataset/train/images`、`dataset/train/labels`、`dataset/val/images`、`dataset/val/labels`。每行标注格式为归一化的 `类别 cx cy 宽 高`。在 `1_TrainYolov8/dataset/data.yaml` 中配置数据路径和类别名称，也可配置测试集。

```sh
# 使用脚本默认参数训练
python 1_TrainYolov8/yolo_train.py --epochs 100 --batch 32 --imgsz 640 --workers 4 --seed 0

# 评估用于视频处理的模型
python 1_TrainYolov8/yolo_evaluate.py --weights 2_GetResults/best.pt --split val

# 可选的测试集评估
python 1_TrainYolov8/yolo_evaluate.py --weights 2_GetResults/best.pt --split test
```

训练还支持 `--model`、`--optimizer`、`--lr0`。默认 `--optimizer auto` 自动选择优化器和学习率；手动指定时可使用 `--optimizer AdamW --lr0 0.001`。

两个训练入口都会在启动训练时直接删除旧的 `1_TrainYolov8/runs/train/`，不保留备份，再将新结果固定保存到该目录。最优权重为 `1_TrainYolov8/runs/train/weights/best.pt`，可将该路径用于尺寸测量和速度提取的 `--weights`。评估脚本输出匹配框的中心/尺寸误差和 IoU。`yolo_train.ipynb` 提供交互式训练和检测演示，从 `1_TrainYolov8/` 目录打开并按需调整输入路径；只有训练单元会清空旧结果，评估和演示单元不会清空。

## 7. 输出与计算设置

| 文件 | 内容 |
| --- | --- |
| `ball_size_measurements/<工况>_ball_sizes.xlsx` | 检测到的球框宽高，单位为像素 |
| `processed_data/ball_sizes_means.csv` | 各工况平均框尺寸，用于比例标定 |
| `processed_data/vx_vy_data_<工况>.csv` | `frame,timestamp,obj_id,vx,vy`；时间戳单位为秒，速度单位为 px/s |
| `processed_data/frame_counts_<工况>.csv` | 逐帧检测/筛选/跟踪/速度数量，以及容器有效状态 |
| `processed_data/frame_counts_<工况>.json` | 数量汇总、筛选配置和完整筛除明细 |

上述两个输出目录均位于 `2_GetResults/` 中。速度 CSV 的 `obj_id` 为 SORT 轨迹编号，`container` 行表示容器速度。同名视频再次运行会覆盖对应输出。

诊断 JSON 的 `ball_filtering` 汇总两类规则各自剔除的框数和涉及帧数。`ball_filter_events` 保存各筛除帧的全部球框、尺寸、置信度、保留/剔除标记及原因，并记录最低保留置信度、全部剔除置信度和数量筛选的分界差距。无事件时该数组为 `[]`。统计单位为跨帧累计的检测框次。诊断 CSV 同时保存筛选前、尺寸筛选后及最终数量，以及每帧实际使用的尺寸阈值。

### 运动计算与物理单位

由于本实验视频采用竖屏拍摄，尺寸测量和速度提取均将解码后的图像逆时针旋转 90 度，使画面恢复正向。旋转后的图像 x 轴向右、y 轴向下。球速度根据连续两帧的 SORT 框中心位移及实际时间戳计算，缺帧时中断该帧间速度计算。容器位置使用 YOLO 初始化和双角点边框掩膜匹配，模板相关分数阈值为 0.85；不可靠位置被省略。容器速度通过连续四个有效位置的线性最小二乘拟合计算，并与球速度的帧间区间对齐；不插值补齐缺失速度。

分析使用的球直径为 0.014 m、质量为 0.0036 kg，各工况分别进行像素到米的换算：

```text
d_px = (mean_width_px + mean_height_px) / 2
scale = 0.014 / d_px
v_m_per_s = v_px_per_s * scale
```

使用不同球体时，同步修改分析脚本中的物理常数。实验室系颗粒温度来自二维 Maxwell 速率分布拟合，速度分量采用零均值 Gaussian 拟合；容器系温度由二阶矩计算，Maxwell/Gaussian 曲线作为参考。`_Moment.py` 脚本提供实验室系二阶矩对照。二阶矩使用测得的速度，不扣除平均速度。

## 许可证

许可证与第三方组件说明见 [LICENSE](LICENSE)、[LICENSES/GPL-3.0.txt](LICENSES/GPL-3.0.txt) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
