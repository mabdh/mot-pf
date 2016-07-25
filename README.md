# Multiple-Object Tracking Using Particle Filter
This is the project for Computer Graphics Lab in University of Bonn
## Implementing a multiple object tracking with Particle Filter based on this reference.
```
M. D. Breitenstein, F. Reichlin, B. Leibe, E. Koller-Meier and L. V. Gool
”Online multi-person tracking-by-detection from a single, uncalibrated
camera”, IEEE Trans. Pattern Anal. Mach. Intell., vol. 33, no. 9, pp.1820
-1833 2011.
```

This is an Eclipse Project.

## Structure of Directory
This directory contains :
- Debug
- Evaluate
> Store evaluation data for every experiment. ex : *Eval1_* is the second experiment. Create folder of EvalN_ manually if you want to do N+1 experiments. With data format : 
```
x-axis-FalseNegative-FalsePositive-TruePositive-IDSwitch-MOTA-MOTP
```
(could be anything, example: number of particles, sigma propagation, etc)

- PETS_frame
> Contain sequence of video. With filename **format frame_0000.jpg**

- PETS_gt,
> Contain ground truth data with filename format frame_0000.txt. With data format :
```
number_of_frame-x(top-left)-y(top-left)-width-height-index
```

- PETS_GT_map
> Contain detection maps from ground truth

- PETS_map
> Contain detection maps from detectors
- src
- TrackerData
> Contain ground truth data with filename format frame_0000.txt. With data format : 
 ```
 number_of_frame-x(top-left)-y(top-left)-width-height-index
 ```
- View1_result
> Result of tracked objects on the image

## Note 
There is also 
- a documentation inside src/html/index.html
- A python file to plot the evaluation
	- EvaluateNumParticles (To evaluate tracker with x-axis is number of particles)
	- EvaluatePETS (To evaluate tracker with x-axis is sigma_of_propagation (could be changed..))

##Dependencies
In Eclipse project. Go to
```
Project->Properties->C/C++ Build->Settings
```

In GCC C++
```
Compiler->Miscelanneous->add other flags "-std=c++11 -std=c++0x"
```

In C++ 
```
Linker->Libraries->add Libraries (-l) "config++"
```


##Configuration File Structure

```
particlefilter :
{
    main :
    {
        num_experiment = 1;									// number experiment that will be executed with various parameters
        detmapfiles = "PETS_map/";							// directory of detection maps
        trackerdatafiles = "TrackerData/";					// directory of tracker data *.txt
        groundtruthfiles = "PETS_gt/";						// directory of ground truth data *.txt
        evalresultfiles = "Evaluate/";						// directory to store evaluation result
        sequenceframefiles = "PETS_frame/";					// directory of sequence frames
        resulttrackerfiles = "View1_result/";				// directory to store image of tracked objects
        main0 :												// first experiment
        {
            min_area_detection_threshold = 30;				// threshold value to find contour of objects
            number_of_particles = 50;						// number of particles
            threshold_detection = 160;						// threshold value to treshold binary images
            use_histogram = 3;								// tipe of histograms (single, divide into 3, overlap)
            vis_type = 2;									// visualization type (particles, circle, bounding box)
            with_gt = 1;									// flag status to use ground truth or not
            sigma_propagate = [10.0, 10.0, 1.0, 1.0];		// value of sigma propagation (sigma_p and sigma_v)
            sigma_measurement = 1.0;						// value of sigma measurement
            c_detection = 0.6;								// constant to weight detection model (1-alpha)
            c_color = 0.4;									// constant to weight color (alpha)
            mul_factor_c_color = 0.6;						// constant to weight detection model (1-alpha),  when there is an occlusion
            mul_factor_c_detection = 0.4;					// constant to weight color (alpha), when there is an occlusion
            width_default = 40;								// width default of bounding box
            height_default = 80;							// height default of bounding box
            using_occlusion_handling = 1;					// flag status to run using occlusion handling or not
        };
        main1 :												// here comes another variation of parameter
        {
        ...
        };
        main2 :												// here again..
        ...
        ...
        ...
    };
}
```

The documentation of ParticleFilter class can be found here [Particle Filter Class](https://mabdh.github.io/Particle-Filter/classParticleFilter.html)

The report can be found here [Project Report](https://github.com/mabdh/Particle-Filter/blob/master/pdf/LabReportv2.pdf)
