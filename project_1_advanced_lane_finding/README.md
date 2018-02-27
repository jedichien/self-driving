# Advanced Lane Finding
Thank [ndrplz] from great tutorial, then I reconstruct it into more easier comprehension. This tutorial only use simple Computer Vision theory to process, however it can be optimised in more Intelligent (s.t. CNN).

<p aling='center'>
<img src='./output/result.gif' width='50%' height='50%'/><br/>
Final view
</p>

## Process concept
1. Camera calibration. see [1_calibration.ipynb]('./1_calibration.ipynb')
2. Finding lane contour. see [2_binarization.ipynb]('./2_binarization.ipynb')
2. Transform image into perspective view which is so-called 'birdeye'. see [3_birdeye.ipynb]('./3_birdeye.ipynb')
4. Calculate polynomial for lane curvature. see [4_lane.ipynb]('./4_lane.ipynb')
5. Main. see [5_main.ipynb](5_main.ipynb)

### Warning
I notice that sometimes will face with np.inf while getting meter from fitted curvature of lane.

[ndrplz]: (https://github.com/ndrplz/self-driving-car/tree/master/project_4_advanced_lane_finding)
