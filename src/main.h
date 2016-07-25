//
//  main.h
//  Particle
//
//  Created by Muhammad Abduh on 18/05/15.
//  Copyright (c) 2015 Muhammad Abduh. All rights reserved.
//

#ifndef Particle_main_h
#define Particle_main_h

#include "ParticleFilter.h"


void getFrame(unsigned int frame, Mat& img);
void writeNewImage(Mat& image, int frame);
void run_detection(int frame, Mat& detections_mat);
void freeIntMat(int** C, int rows);

Mat HSVTemplate; /*!< Template histogram */
Rect detection; /*!< detection bounding box */
string tracker_data_files; /*!< directory of tracker data (*.txt) */
string detmap_files; /*!< directory of detection maps */
string ground_truth_files; /*!< directory of ground truth data (*.txt) */
string eval_result_files; /*!< directory to store evaluation result */
string sequence_frame_files; /*!< directory contain sequence images */
string result_tracker_files; /*!< directory to store images with tracked objects */
char filename[30];

//it is assigned only for initialization
int number_of_particles = 100;
int min_area_detection_threshold = 30; /*!< Threshold to obtain contour of detection map*/
int threshold_detection = 160;  /*!< Thresholding the binary image with this threshold*/
int threshold_detection_gt = 160; /*!< Threshold value if detection map is ground truth*/
int threshold_detection_detector = 160; /*!< Thresholding value if detection map is from detector*/
int use_histogram = 10; /*!< flag status to use divided area of histogram, overlapped, or use histogram as it is*/
int num_experiment = 1; /*!< number of experiment*/

int vis_type = 3; /*!< visualization type */

//Evaluation
int num_files; /*!< number of frames that will be compared*/
double eval_intersect_threshold = 0.2; /*!< Intersection threshold to determine true positive*/
double MOTP_threshold = 0.5; /*!< threshold to determine MOTP*/
int with_gt=0; /*!< flag status whether detection map is from ground truth or not*/
int with_occlusion_handling=1; /*!< flag status to do tracking using occlusion handling*/

double sigma_propagate_[NUM_STATES]={0.0, 0.0, 0.0, 0.0};
double sigma_measurement_ = 5.0;
double c_det_ = 1.0;	/*!< Constant for detection model */
double c_col_ = 1.0; /*!< Constant for color model */
double m_c_det_ = 1.0; /*!< Constant for detection model if there is an occlusion */
double m_c_col_ = 1.0; /*!< Constant for color model if there is an occlusion */
int _height_ = 100; /*!< Bounding box height default */
int _width_ = 80; /*!< Bounding box width default */
#endif
