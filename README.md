# OpenMeanShift
Speeded up version of Mean Shift segmentation based on implemetation in [EDISON system](http://coewww.rutgers.edu/riul/research/code/EDISON/).

Two new speedups implemented:
 - [Multhreaded](/edison_gpu/segm/tdef.h#L49) version for multicore CPU
 - [OpenCL](/edison_gpu/segm/tdef.h#L50) version for GPU
 - [AUTO](/edison_gpu/segm/tdef.h#L51) version for workload distribution between all GPUs and CPU 
 
Results of mean shift segmentation with all versions are very close to results of [NO_SPEEDUP](/edison_gpu/segm/tdef.h#L46) implemetation in EDISON system (difference is negligible and caused by floating point error).

Also regions fusion algorithm speeded up: linked lists replaced with vectors + multithreaded approach. 

To enable original version of regions fusion you can replace [```VANILLA_VERSION 0```](/edison_gpu/segm/msImageProcessor.cpp#L2130) with ```VANILLA_VERSION 1```.

# How to use
```
git clone https://github.com/PolarNick239/OpenMeanShift
cd OpenMeanShift
git submodule update --init
mkdir build
cd build
cmake ..
make -j4
segmentation_demo/segmentation_demo ../data/unicorn_512.png unicorn_segmentation.jpg
```

If you want to use CPU-only or single GPU version instead of auto distributing between all GPUs and CPU - replace [```AUTO_SPEEDUP```](/segmentation_demo/src/main.cpp#L26) with ```MULTITHREADED_SPEEDUP``` or ```GPU_SPEEDUP```.

# Example results
| Input image              | HIGH_SPEEDUP (original EDISON)  | NO_SPEEDUP (original EDISON) |
:-------------------------:|:-------------------------------:|:-----------------------------:
![Unicorn original](/data/unicorn_512.png?raw=true) | ![Unicorn HIGH_SPEEDUP](/data/high_speedup/unicorn_512_high.png?raw=true) | ![Unicorn NO_SPEEDUP](/data/no_speedup/unicorn_512_no.png?raw=true)
![2K texture original](/data/eastern_tower_2048.jpg?raw=true) | ![2K texture HIGH_SPEEDUP](/data/high_speedup/eastern_tower_2048_high.jpg?raw=true) | ![2K texture NO_SPEEDUP](/data/no_speedup/eastern_tower_2048_no.jpg?raw=true)

# Benchmarking
Benchmarking done for 2048x2048 RGB [image](/data/eastern_tower_2048.jpg):

<table>
  <tr>
    <td colspan="3"><b>Mean shift filter</b></td>
    <td colspan="3"><b>+ Regions fusion</b></td>
    <td><b>= Total</b></td>
  </tr>
  <tr>
    <td>Method</td>
    <td>Device</td>
    <td>Time</td>
    <td>Method</td>
    <td>Device</td>
    <td>Time</td>
    <td>Time</td>
  </tr>
  <tr>
    <td>Original <a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L48>HIGH_SPEEDUP</a></td>
    <td>i7 6700</td>
    <td>136 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=1</a></td>
    <td>i7 6700</td>
    <td>21 s</td>
    <td>157 s</td>
  </tr>
  <tr>
    <td>Original <a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L48>HIGH_SPEEDUP</a></td>
    <td>i7 5960X</td>
    <td>370 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=1</a></td>
    <td>i7 5960X</td>
    <td>32 s</td>
    <td>402 s</td>
  </tr>
  <tr>
    <td>Original <a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L46>NO_SPEEDUP</a></td>
    <td>i7 6700</td>
    <td>145 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=1</a></td>
    <td>i7 6700</td>
    <td>7.3 s</td>
    <td>152 s</td>
  </tr>
  <tr>
    <td>Original <a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L46>NO_SPEEDUP</a></td>
    <td>i7 5960X</td>
    <td>161 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=1</a></td>
    <td>i7 5960X</td>
    <td>10 s</td>
    <td>171 s</td>
  </tr>
  <tr>
    <td>Original <a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L46>NO_SPEEDUP</a></td>
    <td>i7 6700</td>
    <td>145 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=0</a></td>
    <td>i7 6700</td>
    <td>2.0 s</td>
    <td>147 s</td>
  </tr>
  <tr>
    <td>Original <a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L46>NO_SPEEDUP</a></td>
    <td>i7 5960X</td>
    <td>161 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=0</a></td>
    <td>i7 5960X</td>
    <td>2.0 s</td>
    <td>163 s</td>
  </tr>
  <tr>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L49>MULTITHREADED_SPEEDUP</a></td>
    <td>i7 6700</td>
    <td>50 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=0</a></td>
    <td>i7 6700</td>
    <td>2.0 s</td>
    <td><b>52 s</b></td>
  </tr>
  <tr>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L49>MULTITHREADED_SPEEDUP</a></td>
    <td>i7 5960X</td>
    <td>22 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=0</a></td>
    <td>i7 5960X</td>
    <td>2.0 s</td>
    <td><b>24 s</b></td>
  </tr>
  <tr>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L50>GPU_SPEEDUP</a></td>
    <td>Titan X (Maxwell)</td>
    <td>6.8 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=0</a></td>
    <td>i7 5960X</td>
    <td>2.0 s</td>
    <td><b>8.8 s</b></td>
  </tr>
  <tr>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L50>GPU_SPEEDUP</a></td>
    <td>R9 390X</td>
    <td>6.3 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=0</a></td>
    <td>i7 6700</td>
    <td>2.0 s</td>
    <td><b>8.3 s</b></td>
  </tr>  
  <tr>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/tdef.h#L50>GPU_SPEEDUP</a></td>
    <td>GTX 1080</td>
    <td>4.0 s</td>
    <td><a href=https://github.com/PolarNick239/OpenMeanShift/blob/master/edison_gpu/segm/msImageProcessor.cpp#L2130>VANILLA_VERSION=0</a></td>
    <td>i7 5960X</td>
    <td>2.0 s</td>
    <td><b>6.0 s</b></td>
  </tr>
</table>

