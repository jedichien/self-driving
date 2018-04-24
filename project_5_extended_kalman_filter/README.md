# Extended Kalman Filter Project
Self-Driving Car Engineer Nanodegree Program

<p align='center'>
<img src='./output_img/demo_dataset1.gif' style='width:80%;height:80%' /><br/>
DATASET 1
</p>

<p align='center'>
<img src='./output_img/demo_dataset2.gif' style='width:80%;height:80%' /><br/>
DATASET 2
</p>

In this project you will utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower than the tolerance outlined in the project rubric. 

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see [this concept in the classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77) for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

Note that the programs that need to be written to accomplish the project are src/FusionEKF.cpp, src/FusionEKF.h, kalman_filter.cpp, kalman_filter.h, tools.cpp, and tools.h

The program main.cpp has already been filled out, but feel free to modify it.

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.


INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)


OUTPUT: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

---

## Other Important Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make` 
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./ExtendedKF `

## Generating Additional Data

This is optional!

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.


## Brief Introduction
Kalman Filter aims to correct noise. The sensor is not always perfect enough to get no error data, however how to get a better accuracy is the major problem.<br/>

To solve this problem, we can apply some of Filters to it, in this case we use Kalman Filter to solve.<br/>

Kalman Filter can handle the single distribution environment which means noise only from the single distribution, but Extended Kalman Filter can handle dynamic case.
For convenient to explain, we suppose the noises consist of normal distribution.<br/>

When we collect data from sensor, Radar and Lidar, we make a estimation for the future, however, if our estimation is pretty bad, how should be improved.<br/>
To improve our estimation, we have to apply a correcting mechanism on it, Kalman Filter.<br/>

Kalman Filter is a accurating matrix which will be changed period, and Extended Kalman Filter(EKF) differ to Kalman Filter is that EKF consider dynamic distribution case.
<p align='center'>
<img src='./output_img/equation_kf.png' /><br/>
Kalman Filter equation
</p>

<hr style='width:50%;'/>

<p align='center'>
<img src='./output_img/equation_ekf.png' /><br/>
Extended Kalman Filter equation
</p>

<hr style='width:50%;'/>

<p align='center'>
<img src='./output_img/matrix_measurement.png' /><br/>
Matrix of H in EKF
</p>

## Reference
1. blog: [Blogger](https://medium.com/intro-to-artificial-intelligence/extended-kalman-filter-simplified-udacitys-self-driving-car-nanodegree-46d952fce7a3)
2. code: [NikolasEnt](https://github.com/NikolasEnt/Extended-Kalman-Filter)
3. code: [ndrplz](https://github.com/ndrplz/self-driving-car/tree/master/project_6_extended_kalman_filter)


