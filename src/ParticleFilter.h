/*! ParticleFilter.h */
//
//  ParticleFilter.h
//  Workspace
//
//  Created by Muhammad Abduh on 10/05/15.
//  Copyright (c) 2015 Muhammad Abduh. All rights reserved.
//

#ifndef __Workspace__ParticleFilter__
#define __Workspace__ParticleFilter__

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <opencv2/opencv.hpp>
#include <libconfig.h++>

#define NUM_STATES 4
#define PI 3.14

using namespace std;
using namespace cv;
using namespace libconfig;

enum STATE_NAME{
    X_POS=0,
    Y_POS,
    X_VEL,
    Y_VEL,
    X_ACC,
    Y_ACC,
};

class ParticleFilter{
    
public:
    ParticleFilter(Mat frame0=Mat(0,0,0), unsigned int n_particles=0, int detidx=0, int frame=0, Rect detectionBB=Rect(0,0,0,0), int use_hist=0, int imgcols=0, int imgrows=0);
    void getMainData(double sigma_propagate_[NUM_STATES], double const& sigma_measurement_, double const& c_detection_, double const& c_color_, double const& mul_factor_c_color_, double const& mul_factor_c_detection_, int const& width_default_, int const& height_default_);
    
    int read_configuration(void);
    void initialization(Rect detection);
    
    bool update(Mat& img,Mat& img_prev,Rect& detection_new, int use_hist, int frame_num);
    unsigned int getNumParticles(void){return numParticles;}
    Mat_<double> getParticles(void){return particles;}
    vector<double> getWeight(void){return weight;}
    void prediction(Mat& image);
    void measurement(Mat_<double>& particles, Rect detection_new, Mat HSVinterpolated, vector<Mat>& MultiHSVInterpolated, Mat& image, int use_hist, int frame);
    int resampling(void);
    void printParticles(Mat_<double>& particles);
    void printWeight(void);
    vector<Point> &getPointTarget(void);
    void writeNewImage(Mat& image,int frame, Rect detection_new);
    void computeHSVHist(const Mat src, Mat& outHist);
    void computeHistOrientation(const Mat& src, Mat& outHist);
     bool isInBoundary(Rect r, Mat image);
    void systematicResampling(vector<double>& w, vector<int>& indx);
    
    void HSVInterp(Mat HSV1, Mat HSV2, Mat& res);
    int getDetectionIndex(void);
    Point getMeanPosition(void);
    
    int getTrackerWidth(void);
    int getTrackerHeight(void);
    
    void setRelyDetection(bool rd);
    bool getRelyDetection(void);
    int getTimesOccluded(void);
    void resetTimesOccluded(void);
    void incTimesOccluded(void);
protected:
    
    Mat_<double> particles; /*!< Location of particles before propagated */
    Mat_<double> particles_temp;
    Mat_<double> particles_new; /*!< Location of new particles after propagated*/
    Point_<int> PFTrack; /*!< Tracker position in current frame */
    Point_<int> PFTrackPrev; /*!< Tracker position in previous frame */
    Point_<double> PFTrackVel; /*!< Tracker velocity in current frame*/
    vector<double> weight; /*!< vector to store weight of particles */
    vector<double> cumulative; /*!< vector to store cumulative normalized weight of particles */
    vector<Point_<int> > point_for_particles; /*!< Location of particles in a tracker in Point datatype */
    
    unsigned int numParticles;

    Rect detectionprev; /*!< Store previous data detection */
    Mat HSVTemplate; /*!< Store histogram of object template */
    Mat imageTemplate; /*!< Store object template */
    Rect detection;
    char filename[30];
    int detection_index = 0;
    int width_template;
    int height_template;
    int width_tracker;
    int height_tracker;
    vector<Rect> trackerbox; /*!< Store tracker bounding box */
    
    vector<Mat> MultiHSVTemplate; /*!< array to store 3 regions of template histogram */
    bool rely_detection; // if true, then use detection in measurement, if false, then just predict mean position with mean velocity
    
    double sigma_propagate[NUM_STATES]={0.0, 0.0, 0.0, 0.0};
    double sigma_measurement = 5.0;
    double c_det = 1.0;	/*!< Constant for detection model */
    double c_col = 1.0; /*!< Constant for color model */
    double m_c_det = 1.0; /*!< Constant for detection model if there is an occlusion */
    double m_c_col = 1.0; /*!< Constant for color model if there is an occlusion */
    int height_ = 100;  /*!< Bounding box height default */
    int width_ = 80;  /*!< Bounding box width default */
    vector<Point> PFTrackSeries; /*!< Array to store last 4 frames tracker position */
    int times_occluded; /*!< store how many frames the occlusion has been occurred */

};

#endif /* defined(__Workspace__ParticleFilter__) */
