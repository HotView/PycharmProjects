# 对极几何

对于旋转矩阵为 $`R_1`$，位置为 $`C_1`$，内在矩阵（intrinsic matrix）为 $`K_1`$ 的相机 1，旋转矩阵为 $`R_2`$，位置为 $`C_2`$，内在矩阵为 $`K_2`$ 的相机 2，

```math
z_{c1}
\begin{bmatrix}
   x_{i1} \\
   y_{i1} \\
   1
\end{bmatrix}
=
K_1 R_1
\left(
\begin{bmatrix}
   x_{o} \\
   y_{o} \\
   z_{o}
\end{bmatrix}
- C_1
\right)
```

```math
z_{c2}
\begin{bmatrix}
   x_{i2} \\
   y_{i2} \\
   1
\end{bmatrix}
=
K_2 R_2
\left(
\begin{bmatrix}
   x_{o} \\
   y_{o} \\
   z_{o}
\end{bmatrix}
- C_2
\right)
```

满足对极约束

```math
\begin{bmatrix}
   x_{i2} & y_{i2} & 1
\end{bmatrix}
F
\begin{bmatrix}
   x_{i1} \\
   y_{i1} \\
   1
\end{bmatrix}
=
\begin{bmatrix}
   x_{i2} & y_{i2} & 1
\end{bmatrix}
K_2^{-T} E K_1^{-1}
\begin{bmatrix}
   x_{i1} \\
   y_{i1} \\
   1
\end{bmatrix}
= 0
```

本质矩阵（essential matrix）为

```math
E = u \times R_2 R_1^T
```

基本矩阵（fundamental matrix）为

```math
F = K_2^{-T} E K_1^{-1}
```

其中

```math
u = R_2 \left( C_1 - C_2 \right)
```
