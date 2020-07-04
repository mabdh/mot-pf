/*! ParticleFilter.cpp */
//
//  ParticleFilter.cpp
//  Workspace
//
//  Created by Muhammad Abduh on 10/05/15.
//  Copyright (c) 2015 Muhammad Abduh. All rights reserved.
//

#include "ParticleFilter.h"


ParticleFilter::ParticleFilter(Mat frame0, unsigned int n_particles, int detidx, int frame, Rect detectionBB, int use_hist, int imgcols, int imgrows){
//! A constructor for ParticleFilter class.
/*!
 This is constructor for ParticleFilter Class to construct a tracker.
 frame0 is image matrix from getFrame(..) function.
 detidx is index of detection
 frame is current number of frame
 detectionBB is bounding box for a detection
 use_hist = 1, calculate color appearance model without dividing it into 3 regions
 use_hist = 2, calculate color appearance model by dividing it into 3 regions
 use_hist = 3, calculate color appearance model by dividing it into 3 regions and overlap each other
 imgcols is width of frame image
 imgrows is height of frame image
*/
    // The initialization of variables
    numParticles = n_particles;
    detection_index = detidx;
    PFTrack = Point_<int>(0,0);
    rely_detection = true;
    particles = Mat_<double>::zeros(numParticles, NUM_STATES);
    particles_new = Mat_<double>::zeros(numParticles, NUM_STATES);
    particles_temp = Mat_<double>::zeros(numParticles, NUM_STATES);
    float init = 1.0/numParticles;
    
    for (unsigned int i=0; i<numParticles; i++){
        weight.push_back(init);
        cumulative.push_back(init);
        point_for_particles.push_back(Point(0,0));
    }

    cumulative.push_back(init);
    setRelyDetection(true);

    // get first reference of detection
    // used for comparing histogram
    //for color appearance model
    
    trackerbox = vector<Rect>(4);
    
    for (unsigned int i = 0; i < trackerbox.size(); i++) {
        trackerbox.at(i) = detectionBB;
    }
    
    //make bounding box for measurement uniform
    int  xfordet = detectionBB.x + (int)(detectionBB.width/2) - (int)(width_/2);
    int  yfordet = detectionBB.y + (int)(detectionBB.height/2) - (int)(height_/2);
    Rect bbDet(xfordet,yfordet,width_,height_);
    
    if (bbDet.x < 0){
        bbDet.x = 0;
    }
    if(bbDet.x+bbDet.width > imgcols){
        bbDet.width = imgcols - bbDet.x;
    }
    if(bbDet.y < 0){
        bbDet.y = 0;
    }
    if(bbDet.y+bbDet.height > imgrows){
        bbDet.height = imgrows - bbDet.y;
    }

    Mat crop0 = frame0(bbDet);
    bool debug_app = 0;
    imageTemplate = crop0;
    if (use_hist == 1) { // divide into one region
        cout << crop0.channels() << endl;
        computeHSVHist(crop0, HSVTemplate);
    }
    else if (use_hist == 2){ // divide into three regions
        int third = crop0.rows/3;
        Mat temp, hsvtemp;
        temp = crop0.rowRange(0, third);
        computeHSVHist(temp, hsvtemp);
        MultiHSVTemplate.push_back(hsvtemp);
        temp = crop0.rowRange(third, third+third);
        computeHSVHist(temp, hsvtemp);
        MultiHSVTemplate.push_back(hsvtemp);
        temp = crop0.rowRange(third+third, crop0.rows);
        computeHSVHist(temp, hsvtemp);
        MultiHSVTemplate.push_back(hsvtemp);
        hsvtemp.release();
        temp.release();
    }
    else if (use_hist == 3){ // divide into three regions and overlap
        int third = crop0.rows/3;
        int half = crop0.rows/2;
        Mat temp, hsvtemp;
        if(debug_app==1){
            imshow("t0", crop0);
            cvWaitKey();
        }
        temp = crop0.rowRange(0, third);
        if(debug_app==1){
            imshow("t1", temp);
            cvWaitKey();
        }
        computeHSVHist(temp, hsvtemp);
        normalize(hsvtemp, hsvtemp, 0, hsvtemp.rows, NORM_MINMAX, -1, Mat() );

        MultiHSVTemplate.push_back(hsvtemp);
        temp = crop0.rowRange(third, half+third);
        
        if(debug_app==1){
            imshow("t2", temp);
            cvWaitKey();
        }
        computeHSVHist(temp, hsvtemp);
        normalize(hsvtemp, hsvtemp, 0, hsvtemp.rows, NORM_MINMAX, -1, Mat() );

        MultiHSVTemplate.push_back(hsvtemp);
        temp = crop0.rowRange(half, crop0.rows);
        if(debug_app==1){
            imshow("t3", temp);
            cvWaitKey();
        }
        computeHSVHist(temp, hsvtemp);
        normalize(hsvtemp, hsvtemp, 0, hsvtemp.rows, NORM_MINMAX, -1, Mat() );
        
        MultiHSVTemplate.push_back(hsvtemp);
        hsvtemp.release();
        temp.release();
    }

    times_occluded = 0;
    width_template = detectionBB.width;
    height_template = detectionBB.height;
    width_tracker = detectionBB.width;
    height_tracker = detectionBB.height;

    // some initialization again in this function
    initialization(detectionBB);
}

void ParticleFilter::getMainData(double sigma_propagate_[NUM_STATES], double const& sigma_measurement_, double const& c_detection_, double const& c_color_, double const& mul_factor_c_color_, double const& mul_factor_c_detection_, int const& width_default_, int const& height_default_){
//! Function to get configuration data.
/*!
 Get the data from the configuration file which is opened in main file
 All of the arguments are the data that are needed by this class, but passed from configuration file, through the main file
*/
	for (int i = 0; i < NUM_STATES; i++) {
        sigma_propagate[i] = sigma_propagate_[i];
    }
    sigma_measurement = sigma_measurement_;
    c_det = c_detection_;
    c_col = c_color_;
    m_c_col = mul_factor_c_color_;
    m_c_det = mul_factor_c_detection_;
    width_ = width_default_;
    height_ = height_default_;
}


void ParticleFilter::printParticles(Mat_<double>& particles){
//! Function to print and debug particles.
/*!
	Run this function to know where the particles are. Use it when debug the code.
*/
	for (unsigned int i=0; i<numParticles; i++) {
        
        for (int j=0; j<NUM_STATES; j++) {
            printf("%f ", particles[i][j]);
        }
        printf("\n");
    }
}


void ParticleFilter::printWeight(void){
//! Function to print weight of every particle.
/*!
	Run this function to know the value of weight of every particle. Use it when debug the code.
*/
    for (unsigned int i=0; i<numParticles; i++) {
        printf("\n");
        printf("%f ", weight.at(i));
    }
}


int ParticleFilter::getDetectionIndex(void){
//! Function to get index of tracker.
/*!
	This function will return index of tracker.
*/
	return detection_index;
}

void ParticleFilter::initialization(Rect detection){
//! Function to initialize all variables.
/*!

*/
	// initial state get from detection
    const double init_state[NUM_STATES] = {static_cast<float>(detection.x + detection.width/2), static_cast<float>(detection.y + detection.height/2), 0, 0};
    RNG rng(numParticles);
    // initialize first starter particles & cdf
    for( unsigned int i = 0; i < numParticles; i++ )
    {
        for( unsigned int j = 0; j < NUM_STATES; j++ )
        {
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dis(0, numParticles);
            float rand = rng.gaussian(sigma_propagate[j]);
            particles[i][j] = init_state[j] + rand;
        }
        // calc cdf
        cumulative.at(i) += i * 1.0 / numParticles;
    }
    cumulative.at(numParticles) = 1.0;
    // initialization of mean value of detection
    PFTrack.x = detection.x + detection.width/2;
    PFTrack.y = detection.y + detection.height/2;
    PFTrackPrev.x = detection.x + detection.width/2;
    PFTrackPrev.y = detection.y + detection.height/2;
    PFTrackVel.x = 0.0;
    PFTrackVel.y = 0.0;
    for (int pf = 0; pf < 4; pf++) {
        PFTrackSeries.push_back(PFTrack);
    }
    detectionprev = detection;
}


void ParticleFilter::setRelyDetection(bool rd){
//! Function to set flag Rely Detection
/*!
*/
	rely_detection = rd;
}

bool ParticleFilter::getRelyDetection(void){
//! Function to get the state of flag Rely Detection
/*!
*/
    return rely_detection;
}

bool ParticleFilter::update(Mat& img, Mat& img_prev, Rect& detection_new, int use_hist, int frame_num){
//! Update step for the tracker after prediction
/*!
	Here, update() function contains measurement step, resampling step, and get the mean value of all particles.
*/
	// generate new particles
    Rect detection;;
    Mat HSVinterpolated;
    vector<Mat> MultiHSVInterpolated(3);
    // bounding box previous frame
    for (unsigned int i = 0; i < trackerbox.size()-1; i++) {
        trackerbox.at(i) = trackerbox.at(i+1);
    }
    // bounding box current frame
    trackerbox.at(trackerbox.size()-1) = detection_new;
    // calculate the average width and height of all trackers
    int width_avg=0, height_avg=0;
    for (unsigned int i = 0; i < trackerbox.size(); i++) {
        width_avg += trackerbox[i].width;
        height_avg += trackerbox[i].height;
    }
    width_avg = (int)((double)(width_avg/trackerbox.size()));
    height_avg = (int)((double)(height_avg/trackerbox.size()));
    if (width_avg < width_/2) {
        width_avg = width_;
    }
    if (height_avg < height_/2) {
        height_avg = height_;
    }
    width_tracker = width_avg;
    height_tracker = height_avg;
    // if there is no occlusion
    if (rely_detection) {
        detection = detection_new;
    }
    else{ // if there is occlusion
        detection = detectionprev;
    }
    // size of tracker and template should be same to be compared
    if (width_tracker > width_template) {
        width_tracker = width_template;
    }
    if (height_tracker > height_template) {
        height_tracker = height_template;
    }
    
    Rect crop(PFTrack.x-width_tracker/2,PFTrack.y-height_tracker/2,width_tracker,height_tracker);
    
    if(use_hist==1){ // compute one region
        if ((crop.x < 0) || (crop.x+crop.width > img_prev.cols) || (crop.y < 0) || (crop.y+crop.height > img_prev.rows) ) {
            HSVInterp(HSVTemplate, HSVTemplate, HSVinterpolated);
        }
        else{
            Mat imHSV = img_prev(crop);
           Mat HSVbefore;
            computeHSVHist(imHSV, HSVbefore);
            HSVInterp(HSVbefore, HSVTemplate, HSVinterpolated);
            HSVbefore.release();
       }
    }
    else if (use_hist == 2){// compute three regions
        if ((crop.x < 0) || (crop.x+crop.width > img_prev.cols) || (crop.y < 0) || (crop.y+crop.height > img_prev.rows) ) {
            Mat HSVTemp;
            for (unsigned int i = 0; i < MultiHSVTemplate.size(); i++) {
                HSVInterp(MultiHSVTemplate[i], MultiHSVTemplate[i], HSVTemp);
                MultiHSVInterpolated.at(i) = HSVTemp;
            }
         }
        else
        {
            Mat imHSV = img_prev(crop);
            vector<Mat> MultiIMHSV;
             Mat tempHSV;Mat beforetemp;
            int third = imHSV.rows/3;
            tempHSV = imHSV.rowRange(0, third);
            computeHSVHist(tempHSV, beforetemp);
            MultiIMHSV.push_back(beforetemp);
            tempHSV = imHSV.rowRange(third,third+third);
            computeHSVHist(tempHSV, beforetemp);
            MultiIMHSV.push_back(beforetemp);
            tempHSV = imHSV.rowRange(third, imHSV.rows);
            computeHSVHist(tempHSV, beforetemp);
            MultiIMHSV.push_back(beforetemp);
            tempHSV.release();
            beforetemp.release();
            imHSV.release();
            
            Mat temp;
            for (unsigned int i = 0; i < MultiHSVTemplate.size(); i++) {
                HSVInterp(MultiIMHSV[i], MultiHSVTemplate[i], temp);
                MultiHSVInterpolated.at(i) = temp;
            }
         }
    }
    else if (use_hist == 3){ // compute three regions and overlap
        if ((crop.x < 0) || (crop.x+crop.width > img_prev.cols) || (crop.y < 0) || (crop.y+crop.height > img_prev.rows) ) {
            Mat HSVTemp;
            for (unsigned int i = 0; i < MultiHSVTemplate.size(); i++) {
                HSVInterp(MultiHSVTemplate[i], MultiHSVTemplate[i], HSVTemp);
                MultiHSVInterpolated.at(i) = HSVTemp;
            }
        }
        else
        {
            Mat imHSV = img_prev(crop);
            vector<Mat> MultiIMHSV(3);
             Mat tempHSV;Mat beforetemp;
             int third = imHSV.rows/3;
            int half = imHSV.rows/2;
            tempHSV = imHSV.rowRange(0, half);
            computeHSVHist(tempHSV, beforetemp);
            normalize(beforetemp, beforetemp, 0, beforetemp.rows, NORM_MINMAX, -1, Mat() );
            
            MultiIMHSV.at(0) = beforetemp;
            tempHSV = imHSV.rowRange(third,half+third);
            computeHSVHist(tempHSV, beforetemp);
            normalize(beforetemp, beforetemp, 0, beforetemp.rows, NORM_MINMAX, -1, Mat() );
            
            MultiIMHSV.at(1) = beforetemp;
            tempHSV = imHSV.rowRange(half, imHSV.rows);
            computeHSVHist(tempHSV, beforetemp);
            normalize(beforetemp, beforetemp, 0, beforetemp.rows, NORM_MINMAX, -1, Mat() );
            
            MultiIMHSV.at(2) = beforetemp;
            Mat temp;
            for (unsigned int i = 0; i < MultiHSVTemplate.size(); i++) {
                HSVInterp(MultiIMHSV[i], MultiHSVTemplate[i], temp);
                MultiHSVInterpolated.at(i) = temp;
            }
        }
    }

    measurement(particles_new, detection,HSVinterpolated, MultiHSVInterpolated, img, use_hist, frame_num);
    vector<Mat>().swap(MultiHSVInterpolated);
    // normalize weight
    double w_sum = 0.0;
    for (unsigned int i = 0; i < numParticles; i++) {
        w_sum += weight.at(i);
    }
    // check if all particles are invalid
    if (w_sum == 0.0) {
        cerr << "[Warning] none of the particles is valid. "
        << "the particle filter is reset." << endl;
        initialization(detection);
        return false;
    }
    vector<double> normalized_weight(numParticles,0.0);
    for (unsigned int i = 0; i < numParticles; i++) {
        normalized_weight[i] = weight.at(i)/w_sum;
    }
    //weighted mean pos
    double xpos = 0.0, ypos = 0.0, xvel = 0.0, yvel = 0.0;
    for (unsigned int i = 0; i < numParticles; i++) {
        particles_temp[i][X_POS] = particles_new[i][X_POS] * normalized_weight.at(i);
        particles_temp[i][Y_POS] = particles_new[i][Y_POS] * normalized_weight.at(i);
        particles_temp[i][X_VEL] = particles_new[i][X_VEL] * normalized_weight.at(i);
        particles_temp[i][Y_VEL] = particles_new[i][Y_VEL] * normalized_weight.at(i);
        
        xpos += particles_temp[i][X_POS];
        ypos += particles_temp[i][Y_POS];
        xvel += particles_temp[i][X_VEL];
        yvel += particles_temp[i][Y_VEL];
    }
    detectionprev = detection;
    
    // Array PFTrackSeries will store tracker position from last 4 frames
    for (int pf =  (int)(PFTrackSeries.size()-1); pf >= 1; pf--) {
        PFTrackSeries[pf] = PFTrackSeries[pf-1];
    }
    PFTrackSeries[0] = PFTrack;
    // Assign tracker positiont to the next frame
    PFTrackPrev = PFTrack;
    PFTrack.x = xpos;
    PFTrack.y = ypos;
    // Calculate tracker velocity from position
    PFTrackVel = PFTrack - PFTrackPrev;
    
    // resample particles
    vector<int> indx(numParticles,0);
    for (unsigned int i = 0; i < numParticles; i++) {
        indx.at(i) = i;
    }
    systematicResampling(normalized_weight, indx);
    for (unsigned int i =0; i < numParticles; i++) {
        int id = indx[i];
        for (int j = 0; j < NUM_STATES; j++) {
            particles[i][j] = particles_new[id][j];
        }
    }

    return true;
}

int ParticleFilter::getTrackerWidth(void){
//! Function to get width of tracker bounding box
/*!
*/
	return width_tracker;
}

int ParticleFilter::getTrackerHeight(void){
//! Function to get height of tracker bounding box
/*!
*/
	return height_tracker;
}

void ParticleFilter::prediction( Mat& image){
//! Prediction step of particle filter
/*!
*/
	vector<double> random(NUM_STATES);
    
    random_device rd;
    mt19937 gen(rd());
    for (unsigned int index = 0; index < numParticles; index++) {
    	// Precompute random numbers
        for (int j = 0; j < NUM_STATES; j++) {
            normal_distribution<> d(0,sigma_propagate[j]);
            random[j] = d(gen);
        }
        // Propagating the particles
        particles_new[index][X_POS] = particles[index][X_POS] + particles[index][X_VEL] + random[0];
        particles_new[index][Y_POS] = particles[index][Y_POS] + particles[index][Y_VEL] +  random[1];
        
        if (rely_detection) {
            particles_new[index][X_VEL] = particles[index][X_VEL] + random[2];
            particles_new[index][Y_VEL] = particles[index][Y_VEL] + random[3];
        }
        else{
            particles_new[index][X_VEL] = 0.0;
            particles_new[index][Y_VEL] = 0.0;
        }
        
        // the particles out of range
        int index_next = index + 1;
        RNG rng(numParticles);
        while(particles_new[index][X_POS] < 0 || particles_new[index][X_POS] > image.cols || particles_new[index][Y_POS] < 0 || particles_new[index][Y_POS] > image.rows){
			for (int j = 0; j < NUM_STATES; j++) {
				random[j] = rng.gaussian(sigma_propagate[j]);
			}
			particles_new[index][X_POS] = PFTrack.x + random[0];//particles[index_next][X_POS];// + particles[index_next][X_VEL] + random.at<double>(0);
			particles_new[index][Y_POS] = PFTrack.y + random[1];//particles[index_next][Y_POS];// + particles[index_next][Y_VEL] + random.at<double>(1);
			particles_new[index][X_VEL] = PFTrackVel.x + random[2];//particles[index_next][X_VEL];// + random.at<double>(2);
			particles_new[index][Y_VEL] = PFTrackVel.y + random[3];//particles[index_next][Y_VEL];// + random.at<double>(3);
			index_next++;
            
        }
        Point_<int> point_int(0,0);
        point_int.x = (int)particles[index][X_POS];
        point_int.y = (int)particles[index][Y_POS];
        point_for_particles[index] = point_int;
        
    }
    vector<double>().swap(random);
}

bool ParticleFilter::isInBoundary(Rect r, Mat image){
//! Function to check whether r is in boundary of the image or not
/*!
*/
	int rright= r.x + r.width;
    int rdown = r.y + r.height;
    if (r.x > 0 && rright < image.cols && r.y > 0 && rdown < image.rows) {
        return true;
    }
    else{
        return false;
    }
}

void ParticleFilter::measurement(Mat_<double>& particles, Rect detection_new, Mat HSVinterpolated,vector<Mat>& MultiHSVInterpolated, Mat& image, int use_hist, int frame){
//! Measurement step of particle filter
/*!
*/
	int x;
    int y;
    
    if (rely_detection) { // when no occlusion
        x = (int)(detection_new.x + detection_new.width/2);
        y = (int)(detection_new.y + detection_new.height/2);
    }
    else{	// when there is an occlusion
		x = PFTrack.x + PFTrackVel.x;
		y = PFTrack.y + PFTrackVel.y;
    }
     //detection term
    double w_det;
    vector<double> w_detm_temp(numParticles,0.0);
    vector<double> w_detm(numParticles,0.0);
    double w_sum = 0.0;
    for (unsigned int index = 0; index < numParticles; index++) {
        double x_term = pow(double(particles_new[index][X_POS]-x), 2)/(2*pow(sigma_measurement,2));
        double y_term = pow(double(particles_new[index][Y_POS]-y), 2)/(2*pow(sigma_measurement,2));
        w_det = (1/sqrt(2*PI*pow(sigma_measurement,2)))*exp(-1*(x_term+y_term));
        w_det = w_det + 1e-99;
        //        w_sum += w_det;
        
        w_detm_temp[index]=w_det;
    }
    
    // normalize weight
    for (unsigned int index = 0; index < numParticles; index++) {
        w_sum += w_detm_temp[index];
    }
    for (unsigned int index = 0; index < numParticles; index++) {
        w_detm[index] = w_detm_temp[index]/w_sum;
    }

    
    //appearance term
    double w_col;
    w_sum = 0.0;
    vector<double> w_colm(numParticles,0.0);
    vector<double> w_colm_temp(numParticles,0.0);
    Mat ibox;
    for (unsigned int index = 0; index < numParticles; index++) {
        
    	// get the bounding box size
        int xcbox = particles_new[index][X_POS]-(int)(width_tracker/2);
        int ycbox = particles_new[index][Y_POS]-(int)(height_tracker/2);
        Rect cbox(xcbox,ycbox,width_tracker,height_tracker);
        
        if(use_hist==1) { // only take one regions
            
            if ((cbox.x < 0) || (cbox.x+cbox.width > image.cols) || (cbox.y < 0) || (cbox.y+cbox.height > image.rows) )
            {
               w_col = 0.0000001;
            }
            else
            {
               Mat HSVCalc;
               ibox = image(cbox);
               computeHSVHist(ibox, HSVCalc);
               w_col = compareHist(HSVinterpolated, HSVCalc, CV_COMP_CORREL);
               HSVCalc.release();
                
            }
            w_sum += w_col;
            w_colm_temp.push_back(w_col);
        }
        else if (use_hist==2){ // only take three regions
            
            if ((cbox.x < 0) || (cbox.x+cbox.width > image.cols) || (cbox.y < 0) || (cbox.y+cbox.height > image.rows) )
            {
                w_col = 0.0000001;
            }
            else{
                vector<Mat> HSVCalc;
                ibox = image(cbox);
                vector<Mat> MultiIMHSV;
                Mat tempHSV;
                int third = ibox.rows/3;
                tempHSV = ibox.rowRange(0, third);
                MultiIMHSV.push_back(tempHSV);
                tempHSV = ibox.rowRange(third,third+third);
                MultiIMHSV.push_back(tempHSV);
                tempHSV = ibox.rowRange(third+third, ibox.rows);
                MultiIMHSV.push_back(tempHSV);
                
                for (unsigned int hs = 0; hs < MultiHSVTemplate.size(); hs++) {
                    Mat HSTemp;
                    computeHSVHist(MultiIMHSV[hs], HSTemp);
                    HSVCalc.push_back(HSTemp);
                }
                w_col = 0.0;
                for (unsigned int hs = 0; hs < MultiHSVTemplate.size(); hs++) {
                    int wtemp = compareHist(MultiHSVInterpolated[hs], HSVCalc[hs], CV_COMP_INTERSECT);
                   w_col += wtemp;
                }
           }
            w_sum += w_col;
            w_colm_temp.push_back(w_col);
        }
        else if(use_hist == 3){	// divide into three regions and overlap
            // if out of image bound
            if ((cbox.x < 0) || (cbox.x+cbox.width > image.cols) || (cbox.y < 0) || (cbox.y+cbox.height > image.rows) )
            {
                w_col = 0.0000001;
            }
            else{
                vector<Mat> HSVCalc(3);
                ibox = image(cbox);
                vector<Mat> MultiIMHSV(3);

                int third = ibox.rows/3;
                int half = ibox.rows/2;
                // Partition step
                MultiIMHSV.at(0) = ibox.rowRange(0, half);
                MultiIMHSV.at(1) = ibox.rowRange(third,half+third);
                MultiIMHSV.at(2) = ibox.rowRange(half, ibox.rows);
                // Calculate histogran for every region
                Mat HSTemp;
                for (unsigned int hs = 0; hs < MultiHSVTemplate.size(); hs++) {
                    computeHSVHist(MultiIMHSV[hs], HSTemp);
                    normalize(HSTemp, HSTemp, 0, HSTemp.rows, NORM_MINMAX, -1, Mat() );
                    HSVCalc.at(hs) = HSTemp;
                }
                // calculate the weight by comparing them
                w_col = 0.0;
                for (unsigned int hs = 0; hs < MultiHSVTemplate.size(); hs++) {

                    double wtemp = compareHist(MultiHSVInterpolated[hs], HSVCalc[hs], CV_COMP_BHATTACHARYYA);

                    wtemp = 1.0 - wtemp;
                    w_col += wtemp;
                    w_col /= (int)(MultiHSVTemplate.size());
                }
            }

            w_sum += w_col;
            w_colm_temp.at(index) = w_col;
        }
        else if(use_hist==4){
            
        }
        else{
            w_colm_temp.push_back(0.0);
            w_sum = 1.0;
        }
        
    }
    
    if (!rely_detection) { //dont rely on detection if there is an occlusion
        c_det = m_c_det;
        c_col = m_c_col;
    }
    // normalize weight
    for (unsigned int index = 0; index < numParticles; index++) {

        w_colm.at(index) = w_colm_temp.at(index)/w_sum; //diubah
    }
    //accumulate
    for (unsigned int index = 0; index < numParticles; index++) {
        w_det = w_detm.at(index) * c_det;
        w_detm.at(index) = w_det;
    }
    
    for (unsigned int index = 0; index < numParticles; index++) {
          w_col = w_colm.at(index) * c_col;
        w_colm.at(index) = w_col;

    }
    // total weight
    for (unsigned int index = 0; index < numParticles; index++) {
        weight.at(index) = w_detm.at(index) + w_colm.at(index);
    }
}

vector<Point> &ParticleFilter::getPointTarget(void){
//! Return the tracker position with datatype Point
/*!
*/
	return point_for_particles;
}

int ParticleFilter::getTimesOccluded(void){
//! Return the number of how many frames the tracker occluded
/*!
*/
	return times_occluded;
}

void ParticleFilter::resetTimesOccluded(void){
//! Reset the occlusion time
/*!
*/
	times_occluded = 0;
}

void ParticleFilter::incTimesOccluded(void){
//! Increment times occluded
/*!
*/
	times_occluded++;
}

void ParticleFilter::computeHSVHist(const Mat src, Mat& histdst){
//! Function to compute HSV histogram of image src
/*!
*/
    Mat hsv;
    cvtColor(src, hsv, CV_BGR2HSV);
    
    // Quantize the hue to 12 levels
    // and the saturation to 12 levels
    int hbins = 12, sbins = 12;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};
    calcHist( &hsv, 1, channels, Mat(), histdst, 2, histSize, ranges, true, false );
    hsv.release();
}

void ParticleFilter::systematicResampling(vector<double>& w, vector<int>& indx){
//! Resampling step of particle filter
/*!
*/
	// random from std
    random_device rd;
    mt19937 gen(rd());
    //initialize
    vector<double> Q(numParticles,0.0);
    vector<double> Wnorm(numParticles,0.0);
    // normalize w
    double sum_w = 0.0;
    for (unsigned int i =0 ;  i  < numParticles; i++) {
        sum_w += w.at(i);
    }
    for (unsigned int i =0 ;  i  < numParticles; i++) {
        Wnorm.at(i) = w.at(i) / sum_w;
    }
    // cum sum
    sum_w = 0.0;
    for (unsigned int i =0 ;  i  < numParticles; i++) {
        sum_w += Wnorm.at(i);
        Q.at(i) = sum_w;
    }
    Q.at(numParticles-1)=1.;
    // Uniform random vector
    vector<double> T(numParticles+1);
    uniform_real_distribution<> dis(0, 1);
    for (unsigned int i = 0; i < numParticles; i++) {
        T.at(i) = (double(i) + dis(gen))/numParticles;
        
    }
    T.at(numParticles) = 1;
    // resampling
    unsigned int i = 0, j = 0;
    while (i < numParticles) {
        if (T.at(i) < Q.at(j) && Q.at(j) > 0.4)
        {
            indx.at(i) = j;
            i++;
        }
        else
        {
            j++;
            if (j >= numParticles) {
                j = numParticles - 1;
            }
        }
    }
}

void ParticleFilter::HSVInterp(Mat HSV1, Mat HSV2, Mat& res)
{
//!
/*!
	Interpolate two HSV histogram with weight w
*/
    double w = 0.7;
    multiply(w, HSV1, HSV1);
    multiply((1.0-w), HSV2, HSV2);
    add(HSV1, HSV2, res);
}

Point ParticleFilter::getMeanPosition(void){
//! Get the mean position from all of particles
/*!
	return datatype Point
*/
	Point p;
    p.x = PFTrack.x;
    p.y = PFTrack.y;
    return p;
}
