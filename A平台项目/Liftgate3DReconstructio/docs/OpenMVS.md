# OpenMVS

OpenMVS 程序的基本命令行为：

```
$ 程序名 -i 输入文件 -o 输出文件
```

其实输入文件和输出文件都是 OpenMVS 格式文件（*.mvs）。

## DensifyPointCloud

生成密集点云。

其它命令行参数：

* `--estimate-normals`，为 1 时输出的 PLY 文件包含法线方向（不太准确）；
* `--number-views`，用于估计深度通道的图像数，0 表示使用所有图像，图像越多深度越精确，越少通道越完整。

DensifyPointCloud 算法有缺陷，偶尔会产生 [NaN](https://en.wikipedia.org/wiki/NaN) 点（用 MeshLab 打开产生的点云文件时会提示 “mesh contains 1 vertices with NAN coords”），这不会影响输出的密集点云，但之后的 ReconstructMesh 等环节将无法进行（会在没有任何反馈信息的情况下结束进程）。目前仍不清楚 NaN 点产生的原因。有时只要重新运行一次 DensifyPointCloud，就不再出现 NaN 点。OpenMVG 的 ComputeFeatures 采用 `AKAZE_FLOAT` 或 `AKAZE_MLDB` 算法时似乎更容易出现 NaN 点。

## ReconstructMesh

生成网格模型。

## RefineMesh

优化网格模型。

## TextureMesh

生成材质贴图。

其它命令行参数：

* `--export-type`，设定网格模型文件为 `ply` 或 `obj`，其中 `ply` 是二进制的，文件较小，打开更快，但 Blender 不兼容 `ply` 的材质。
