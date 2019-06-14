# OpenMVG

## sfm_data 文件

OpenMVG 项目文件，可以是二进制（*.bin）或 JSON（*.json）文件。

JSON 文件格式为：

* 根对象
    * `sfm_data_version` 字符串（`0.3`）
    * `root_path` 字符串（默认为 `images`）
    * `views` 数组
        * 成员对象
            * `key` 整型
            * `value` 对象
                * `polymorphic_id` 整型，一般为 1073741824
                * `ptr_wrapper` 对象
                    * `id` 整型
                    * `data` 对象
                        * `id_view` 整型，对应 `views` 成员对象的 `key`
                        * `id_intrinsic` 整型，对应 `intrinsics` 成员对象的 `key`
                        * `id_pose` 整型，对应 `extrinsics` 成员对象的 `key`
                        * `local_path` 字符串，常为空字符串
                        * `filename` 字符串，图像文件名
                        * `width` 整型，图像宽度
                        * `height` 整型，图像高度
    * `intrinsics` 数组
        * 成员对象
            * `key` 整型
            * `value` 对象
                * `polymorphic_id` 整型，一般为 2147483649
                * `polymorphic_name` 字符串，默认为 `pinhole_radial_k3`
                * `ptr_wrapper` 对象
                    * `id` 整型
                    * `data` 对象
                        * `width` 整型，图像宽度
                        * `height` 整型，图像高度
                        * `focal_length` 数值，像素焦距
                        * `principal_point` 数组，主轴坐标，默认为 `[图像宽度/2, 图像高度/2]`
                        * `disto_k3` 数组（`pinhole_radial_k3` 类型时）
    * `extrinsics` 数组
        * 成员对象
            * `key` 整型
            * `value` 对象
                * `rotation` 数组，相机旋转矩阵
                * `center` 数组，相机坐标
    * `structure` 数组
        * 成员对象
            * `key` 整型
            * `value` 对象
                * `X` 数组，三维坐标
                * `observations` 数组
                    * `key` 整型，对应 `views` 成员对象的 `key`
                    * `value` 对象
                        * `id_feat` 整型
                        * `x` 数组，二维特征点坐标
    * `control_points` 数组
        * 成员对象
            * `key` 整型
            * `value` 对象
                * `X` 数组，三维坐标
                * `observations` 数组
                    * `key` 整型，对应 `views` 成员对象的 `key`
                    * `value` 对象
                        * `id_feat` 整型
                        * `x` 数组，二维特征点坐标

## SfMInit_ImageListing

导入图像。

命令行格式：

```
$ openMVG_main_SfMInit_ImageListing -i 输入目录 -o 输出目录 -d 相机数据库
```

或

```
$ openMVG_main_SfMInit_ImageListing -i 输入目录 -o 输出目录 -f 像素焦距
```

* 输入目录：存放照片；
* 输出目录：会在其中生成 sfm_data 文件；
* 相机数据库：一个文本文档，每行记录一种相机（对应 EXIF 的 `Camera Model (0x0110)` 项）的 CCD 传感器宽度信息。
* 像素焦距：对应 `sfm_data.json` 中的 `focal_length`，应该就是 OpenCV 中的 `fx` 和 `fy`。

### 相机数据库

范例：

```
Sony Cybershot DSC TX1;5.9
```

表示 Sony Cybershot DSC TX1 的传感器宽度为 5.9 mm，根据其它资料，这个相机的 CCD 为 1/2.4-inch (~ 5.90 x 4.43 mm)。

传感器宽度用来计算像素焦距，公式为：

```
像素焦距 = max(照片宽度, 照片高度) * 毫米焦距 / 传感器宽度
```

其中毫米焦距即 EXIF 中的 `Focal Length (0x920A)` 项。

## ComputeFeatures

提取二维特征。

命令行格式：

```
$ openMVG_main_ComputeFeatures -i 输入文件 -o 输出目录 -m 算法
```

* 输入文件：即 sfm_data 文件路径；
* 输出目录：会在其中生成 `image_describer.json`、`*.feat` 和 `*.desc`；
* 算法：比如 `SIFT`、`AKAZE_FLOAT`、`AKAZE_MLDB`。

可以用 `*_mask.png` 指定不用于计算 feature 的部分（黑色）。

SfM_Localization 似乎只支持 `SIFT`。

### image_describer.json

### *.feat

四列数值，应该是描述 feature 坐标。

### *.desc

二进制文件，descriptor 数据。

## ComputeMatches

匹配二维特征。

命令行格式：

```
$ openMVG_main_ComputeMatches -i 输入文件 -o 输出目录 -g 几何模型
```

* 输入文件：即 sfm_data 文件路径；
* 输出目录：即 `image_describer.json` 路径，会在其中生成以下文件：
    * `matches.$.bin`（`$` 对应几何模型）
    * `matches.putative.bin`
    * `geometric_matches`
    * `putative_matches`
    * `GeometricAdjacencyMatrix.svg`
    * `PutativeAdjacencyMatrix.svg`
* 几何模型：
    * `f`：Fundamental matrix filtering，默认，IncrementalSfM 需要这个；
    * `e`：Essential matrix filtering，GlobalSfM 需要这个。
    * `h`：Homography matrix filtering。

程序会调用 graphviz（neato）显示结果，但没有也可以正常执行下去。

## IncrementalSfM

估计相机位姿（增量算法）。

命令行格式：

```
$ openMVG_main_IncrementalSfM -i 输入文件 -m 匹配目录 -o 输出目录
```

* 输入文件：即 sfm_data 文件路径；
* 匹配目录：即 ComputeFeatures 和 ComputeMatches 的工作目录；
* 输出目录：会在其中生成以下文件：
    * `sfm_data.bin`
    * `initialPair.ply`
    * `cloud_and_poses.ply`
    * `########_Resection.ply`
    * `residuals_histogram.svg`
    * `SfMReconstruction_Report.html`
    * `Reconstruction_Report.html`

## GlobalSfM

估计相机位姿（全局算法）。

命令行格式：

```
$ openMVG_main_GlobalSfM -i 输入文件 -m 匹配目录 -o 输出目录
```

* 输入文件：即 sfm_data 文件路径；
* 匹配目录：即 ComputeFeatures 和 ComputeMatches 的工作目录；
* 输出目录：会在其中生成以下文件：
    * `sfm_data.bin`
    * `global_relative_rotation_view_graph`
    * `global_relative_rotation_pose_graph`
    * `global_relative_rotation_pose_graph_final`
    * `initial_structure.ply`
    * `structure_##_*.ply`
    * `residuals_histogram.svg`
    * `SfMReconstruction_Report.html`
    * `Reconstruction_Report.html`

## ComputeStructureFromKnownPoses

生成稀疏点云。

命令行格式：

```
$ openMVG_main_ComputeStructureFromKnownPoses -i 输入文件 -m 匹配目录 -o 输出文件
```

* 输入文件：即 sfm_data 文件路径；
* 匹配目录：即 ComputeFeatures 的工作目录；
* 输出文件：生成的 sfm_data 文件路径，还会在同目录下生成以下文件：
    * `*.ply`
    * `residuals_histogram.svg`
    * `SfMStructureFromKnownPoses_Report.html`

如果已知相机位姿，ComputeStructureFromKnownPoses 之前不需要 ComputeMatches。

## ComputeSfM_DataColor

估计点云颜色。

命令行格式：

```
$ openMVG_main_ComputeSfM_DataColor -i 输入文件 -o 输出文件
```

* 输入文件：即 sfm_data 文件路径；
* 输出文件：生成的 PLY 文件路径。

## ConvertSfM_DataFormat

在二进制和 JSON 间转换 sfm_data 文件格式。

命令行格式：

```
$ openMVG_main_ConvertSfM_DataFormat -i 输入文件 -o 输出文件
```

* 输入文件：输入 sfm_data 文件路径；
* 输出文件：输出 sfm_data 文件路径。

## openMVG2openMVS

将 OpenMVG 项目文件（sfm_data 文件）转为 OpenMVG 项目文件（*.mvs）。

命令行格式：

```
$ openMVG_main_openMVG2openMVS -i 输入文件 -o 输出文件
```

* 输入文件：即 sfm_data 文件路径；
* 输出文件：生成的 `*.mvs` 文件路径，作为 OpenMVS 的输入。
