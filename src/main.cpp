/*! main.cpp */
//  main.cpp
//  Workspace
//
//  Created by Muhammad Abduh on 10/05/15.
//  Copyright (c) 2015 Muhammad Abduh. All rights reserved.
//

#include <iostream>
#include "main.h"
#include <hungarian.h>
#include <dirent.h>
#include <fstream>
#include <unistd.h>

int read_configuration_main(void){
//! Read configuration file
/*!
The configuration file is "Parameter.cfg"
This function is used to get the number of experiment that is gonna be executed,
and string for the directory
*/
    Config cfg;
    int numexp;
    // Read the file. If there is an error, report it and exit.
    try
    {
        cfg.readFile("src/Parameter.cfg");
    }
    catch(const FileIOException &fioex)
    {
        std::cerr << "I/O error while reading file main parameter." << std::endl;
        return(EXIT_FAILURE);
    }
    catch(const ParseException &pex)
    {
        std::cerr << "Main : Parse error at " << pex.getFile() << ":" << pex.getLine()
        << " - " << pex.getError() << std::endl;
        return(EXIT_FAILURE);
    }

    const Setting& root = cfg.getRoot();

    try
    {
        const Setting &main_conf = root["particlefilter"]["main"];
        main_conf.lookupValue("num_experiment",numexp);
        main_conf.lookupValue("detmapfiles",detmap_files);
        main_conf.lookupValue("trackerdatafiles",tracker_data_files);
        main_conf.lookupValue("groundtruthfiles",ground_truth_files);
        main_conf.lookupValue("evalresultfiles",eval_result_files);
        main_conf.lookupValue("sequenceframefiles",sequence_frame_files);
        main_conf.lookupValue("resulttrackerfiles",result_tracker_files);

    }
    catch(const SettingNotFoundException &nfex)
    {
    	cerr << "Setting Not Found" << std::endl;
    }
    num_experiment = numexp;
    return(EXIT_SUCCESS);
}

int read_configuration(int main_num){
//! Read configuration file
/*!
The configuration file is "Parameter.cfg"
This function is used to get all of the parameters
*/
    Config cfg;
    int numP;
    int minAreaDet;
    int thDet;
    int uh;
    int vt;
    int gt;
    int oh;
    string strmainnum = "main";
    // Read the file. If there is an error, report it and exit.
    try
    {
        cfg.readFile("src/Parameter.cfg");
    }
    catch(const FileIOException &fioex)
    {
        std::cerr << "I/O error while reading file parameter." << std::endl;
        return(EXIT_FAILURE);
    }
    catch(const ParseException &pex)
    {
        std::cerr << "Main : Parse error at " << pex.getFile() << ":" << pex.getLine()
        << " - " << pex.getError() << std::endl;
        return(EXIT_FAILURE);
    }

    const Setting& root = cfg.getRoot();
    try
    {
        ostringstream out;
        out << strmainnum << main_num;
        const Setting &main_conf = root["particlefilter"]["main"][out.str()];

        main_conf.lookupValue("number_of_particles", numP);
        main_conf.lookupValue("min_area_detection_threshold", minAreaDet);
        main_conf.lookupValue("threshold_detection", thDet);
        main_conf.lookupValue("use_histogram", uh);
        main_conf.lookupValue("vis_type", vt);
        main_conf.lookupValue("with_gt", gt);
        main_conf.lookupValue("using_occlusion_handling",oh);
        unsigned int sp_length =  main_conf["sigma_propagate"].getLength();
        for (unsigned int i = 0; i < sp_length; i++) {
            sigma_propagate_[i] =  (double)(main_conf["sigma_propagate"][i])
            ;
        }
        main_conf.lookupValue("sigma_measurement", sigma_measurement_);
        main_conf.lookupValue("c_detection", c_det_);
        main_conf.lookupValue("c_color", c_col_);
        main_conf.lookupValue("mul_factor_c_color", m_c_col_);
        main_conf.lookupValue("mul_factor_c_detection", m_c_det_);
        main_conf.lookupValue("width_default", _width_);
        main_conf.lookupValue("height_default", _height_);
    }
    catch(const SettingNotFoundException &nfex)
    {
        // Ignore.
    	cout << "SettingNotFoundException" << endl;
    	return(EXIT_FAILURE);
    }
    number_of_particles = numP;
    min_area_detection_threshold = minAreaDet;
    threshold_detection = thDet;
    use_histogram = uh;
    vis_type = vt;
    with_gt = gt;
    with_occlusion_handling = oh;
    return(EXIT_SUCCESS);
}

void getFrame(unsigned int frame, Mat& img){
//! Get the image frame
/*!
Get the image frame from directory "sequence_frame_files".
The file name should have "frame_0000.jpg" format.
frame is number of frame, img is the matrix to place the image that is obtained from this function
*/
    sprintf(filename, "%sframe_%.4d.jpg",sequence_frame_files.c_str(),frame);
    img = imread(filename, CV_LOAD_IMAGE_COLOR);

    if(img.empty())
    {
        cout<<"image not found or read!<<endl; <="" br=""> return -1";
    }
}

void writeNewImage(Mat& image, int frame, int status, ParticleFilter& p, Rect detection, int index){
//! Write an image to a file
/*!
Write the image frame to directory "View1_result".
The file name will have "frame_0000.jpg" format
*/
    int lineType = 8;
    int x,width;
    int y,height;

    x = p.getMeanPosition().x;
    y = p.getMeanPosition().y;
    width = p.getTrackerWidth();
    height = p.getTrackerHeight();

    vector<Point> po;
    po = p.getPointTarget();
    switch (status) {
        case 0:
            //draw mean position as circle
            circle( image,
                   Point(x,y),
                   7,
                   Scalar( (index*0)%60, 0, 255 ),
                   2,
                   lineType );
            break;
        case 1:
            // draw every particle as rect
            for (int j = 0; j < number_of_particles; j++) {
                Scalar color = Scalar( index*50, 255, 0 );
                Rect box(po.at(j).x-width/2, po.at(j).y - height/2, width, height);
                rectangle( image, box, color, 2, 8, 0 );
            }
            break;
        case 2:
        {
            // draw every particle as circle
            char text[5];
            sprintf(text, "%d", p.getDetectionIndex());
            Point textOrg(x, y - detection.height/2 - 5);
            double fontScale = 0.5;
            int thickness = 2;
            int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
            putText(image, text, textOrg, fontFace, fontScale,
                    Scalar::all(255), thickness, 8);

            for (unsigned int j = 0; j < po.size(); j++) {

                circle( image,
                       po[j],
                       2,
                       Scalar( p.getDetectionIndex()*70%255, 70, p.getDetectionIndex()*20%255 ),
                       2,
                       lineType );
            }
            circle( image,
                   p.getMeanPosition(),
                   1,
                   Scalar( 255,255,255),
                   2,
                   lineType );

            rectangle( image, Rect(detection.x+detection.width/2-20/2, detection.y+detection.height/2-50/2, 20, 50),  Scalar( p.getDetectionIndex()*10%255, 70, p.getDetectionIndex()*40%255 ), 2, 2, 0 );
            break;
        }
        default:
            //draw mean position as rect
            Scalar color = Scalar( p.getDetectionIndex()*70%255, 70, p.getDetectionIndex()*20%255 );
            char text[5];
            sprintf(text, "%d", p.getDetectionIndex());
            Point textOrg(x, y - detection.height/2 - 5);
            double fontScale = 0.5;
            int thickness = 2;
            int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
            putText(image, text, textOrg, fontFace, fontScale,
            Scalar::all(255), thickness, 8);
            Rect box(x-width/2, y - height/2, width, height);
            rectangle( image, box, color, 2, 8, 0 );
            break;
    }

    sprintf(filename, "%sframe_%04d.jpg",result_tracker_files.c_str(),frame);
    imwrite(filename, image );
}


void LogTracker(vector<ParticleFilter>* Tr, int frame_num){
//! Save tracker informations
/*!
Store the "frame_num"-th tracker information to directory "tracker_data_files".
These data will be used to evaluate by comparing them with the ground truth.
*/
	ofstream myfile;
    sprintf(filename, "%sTracker_%04d.txt",tracker_data_files.c_str(),frame_num);
    myfile.open (filename);
    for (unsigned int i = 0; i < Tr->size(); i++) {
        int xbb = Tr->at(i).getMeanPosition().x - (Tr->at(i).getTrackerWidth()/2);
        int ybb = Tr->at(i).getMeanPosition().y - (Tr->at(i).getTrackerHeight()/2);
        myfile << xbb << " " << ybb << " " <<  Tr->at(i).getTrackerWidth() << " " <<  Tr->at(i).getTrackerHeight() << " " <<  Tr->at(i).getDetectionIndex() << endl;
    }
    myfile.close();
}

void run_detection(int frame, Mat& detections_mat){
//! Localize objects from detection map
/*!
Localize objects from detection map.
The detection map images are stored in directory "detmap_files".
*/
	Mat threshold_output, src_gray;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    sprintf(filename, "%sframe_%04d.png",detmap_files.c_str(),frame);
    cout<<filename<<endl;
    Mat src = imread( filename, 1  );

    /// Convert image to gray
   cvtColor( src, src_gray, CV_BGR2GRAY );
    /// Detect edges using Threshold
   if (with_gt==1) {
        threshold_detection = threshold_detection_gt;
   }
   else{
        threshold_detection = threshold_detection_detector;
   }
   threshold( src_gray, threshold_output, threshold_detection, 255, THRESH_BINARY );
   src_gray.release();
   // Find contours
   findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
   threshold_output.release();
   // delete contour area < n
   contours.erase(remove_if(contours.begin(), contours.end(),[](vector<Point> p){
                                 return contourArea(p) < min_area_detection_threshold;
                             }), contours.end());
    /// Get the moments
    vector<Moments> mu(contours.size() );
    for( unsigned int i = 0; i < contours.size(); i++ )
    { mu[i] = moments( contours[i], false ); }

    ///  Get the mass centers && score:
    vector<double> score(contours.size());
    vector<Point2f> mc( contours.size() );
    for( unsigned int i = 0; i < contours.size(); i++ )
    { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
        double area = contourArea(contours[i]);
        score[i] = area;
    }
    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );

    for( unsigned int i = 0; i < contours.size(); i++ )
    { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        boundRect[i].x = mc[i].x - boundRect[i].width - (int)boundRect[i].width/2;
        boundRect[i].width = (int) 2*(mc[i].x-boundRect[i].x);
        boundRect[i].y =  mc[i].y - boundRect[i].height - (int)boundRect[i].height/2;
        boundRect[i].height = (int) 2*(mc[i].y-boundRect[i].y);
    }

    detections_mat = Mat::zeros((int)boundRect.size(), 5, CV_32S);
    for (unsigned int i = 0; i < boundRect.size(); i++) {
        detections_mat.at<int>(i,0) = boundRect[i].x;
        detections_mat.at<int>(i,1) = boundRect[i].y;
        detections_mat.at<int>(i,2) = boundRect[i].width;
        detections_mat.at<int>(i,3) = boundRect[i].height;
        detections_mat.at<int>(i,4) = 0;
    }
}

bool isInBoundary(Point p, Mat image, int threshold){
//! Checking whether p is on image boundary or not
/*!
The image boundary is determined by giving the threshold for each side of the image.
*/
	bool res = false;
    if (p.x < threshold) {
        res = true;
    }
    if (p.x > (image.cols - threshold)) {
        res = true;
    }
    if (p.y < threshold) {
        res = true;
    }
    if (p.y > (image.rows - threshold)) {
        res = true;
    }
    return res;
}

double euclideanDist(Point& p, Point& q) {
//! Calculate euclidean distance between two points
/*!
*/
	int difx = p.x - q.x;
    int dify = p.y - q.y;
    return sqrt(difx*difx + dify*dify);
}

Mat_<int> intArrayToMat(int** C, int rows, int cols) {
//! Convert int array to Mat
/*!
Convert a matrix C with datatype int** to datatype Mat_<int>
*/
	Mat_<int> temp(rows,cols,CV_32S);
    int i,j;
    for(i=0; i<rows; i++) {
        for(j=0; j<cols; j++) {
            temp[i][j] = C[i][j];
        }
    }
    return temp;
}

void freeIntMat(int** C, int rows){
//! Used to deallocate allocated int** C from matToIntArray() function.
/*!
*/
	for (int i = 0; i < rows; ++i) {
        free(C[i]);
    }
    free(C);
}

int** matToIntArray(Mat_<int> &m){
//! Convert array with datatype Mat_<int> to datatype int**
/*!
*/
	int i,j;
    int** r;
    r = (int**)calloc(m.rows,sizeof(int*));
    for(i=0;i<m.rows;i++)
    {
        r[i] = (int*)calloc(m.cols,sizeof(int));
        for(j=0;j<m.cols;j++)
            r[i][j] = m[i][j];
    }
    return r;
}

void distanceOcclusionDet(int num_occ,int num_detection, Mat_<int>& distOccDet, Mat& detection_mat, int nomatch_idx, Mat& occ ){
//! Calculate distance between occluded trackers and detection in current frame
/*!
 Output is stored in distOccDet
*/
	//construct distance between detection and previous detection matrix
    distOccDet = Mat_<int>::zeros(num_occ,1);
    for (int i = 0; i < occ.rows; i++) {
        // previous detection is row
        Point curPoint;
        Point prevPoint;
        curPoint.x = detection_mat.at<int>(nomatch_idx,0);
        curPoint.y = detection_mat.at<int>(nomatch_idx,1);
        prevPoint.x = occ.at<int>(i,0);
        prevPoint.y = occ.at<int>(i,1);
        double dist = norm(curPoint - prevPoint);
        distOccDet.at<int>(i,1) = (int)dist;
    }

    hungarian_problem_t occ_pr;
    int** m_detocc = matToIntArray(distOccDet);

    int matrix_size = hungarian_init(&occ_pr, m_detocc , distOccDet.rows,distOccDet.cols, HUNGARIAN_MODE_MINIMIZE_COST) ;
    hungarian_solve(&occ_pr);
    freeIntMat(m_detocc,distOccDet.rows);
    m_detocc = hungarian_get_assignment(&occ_pr);
    distOccDet = intArrayToMat(m_detocc, distOccDet.rows, distOccDet.cols);

    hungarian_free(&occ_pr);
}

void distancePrevAndDet(int num_detection_prev,int num_detection, Mat_<int>& distDetPrev, Mat& detection_mat, Mat& detection_mat_prev ){
//! Calculate distance between detection in previous frame and detection in current frame
/*!
 Output is stored in distDetPrev.
 Previous detection as row
 Current detection as column
*/
	//construct distance between detection and previous detection matrix
    distDetPrev = Mat_<int>::zeros(num_detection_prev,num_detection);
    for (int i = 0; i < detection_mat_prev.rows; i++) {
        // previous detection is row
        for (int j = 0; j < detection_mat.rows; j++) {
            Point curPoint;
            Point prevPoint;
            curPoint.x = detection_mat.at<int>(j,0);
            curPoint.y = detection_mat.at<int>(j,1);
            prevPoint.x = detection_mat_prev.at<int>(i,0);
            prevPoint.y = detection_mat_prev.at<int>(i,1);
            double dist = norm(curPoint - prevPoint);
            distDetPrev.at<int>(i,j) = (int)dist;
        }
    }
    hungarian_problem_t detprev_det_pr;
    int** m_detprev_det = matToIntArray(distDetPrev);


    int matrix_size = hungarian_init(&detprev_det_pr, m_detprev_det , distDetPrev.rows,distDetPrev.cols, HUNGARIAN_MODE_MINIMIZE_COST) ;
    hungarian_solve(&detprev_det_pr);

    freeIntMat(m_detprev_det,distDetPrev.rows);
    m_detprev_det = hungarian_get_assignment(&detprev_det_pr);
    distDetPrev = intArrayToMat(m_detprev_det, distDetPrev.rows, distDetPrev.cols);
    hungarian_free(&detprev_det_pr);
}

void distanceTrackerAndDet(vector<ParticleFilter>& T,int num_detection, Mat_<int>& distTrDetmat, Mat& detection_mat){
//! Calculate distance between trackers and detection in current frame
/*!
 Output is stored in distTrDetmat.
 Trackers as row
 Current detection as column
*/
	//construct distance matrix
    ParticleFilter key_p;
    for (int i = 0; i < (T.size() - 1); ++i){
    	for (int j = 0; j < T.size() - 1 - i; ++j ){
			if (T.at(j).getDetectionIndex() > T.at(j+1).getDetectionIndex()){
				key_p = T.at(j);
				T.at(j) = T.at(j+1);
				T.at(j+1) = key_p;
			}
    	}
    }

    distTrDetmat = Mat_<int>::zeros((int)T.size(),num_detection);
    for (unsigned int tr = 0; tr < T.size(); tr++) {
    	for (int dt = 0; dt < num_detection; dt++) {
            Point detPoint(detection_mat.at<int>(dt,0),detection_mat.at<int>(dt,1));
            Point trackerPoint = T.at(tr).getMeanPosition();
            double dist = norm(trackerPoint - detPoint);
            distTrDetmat.at<int>(tr,dt) = (int)dist;
        }
    }
    hungarian_problem_t track_det_pr;
    int** m_track_det = matToIntArray(distTrDetmat);

    int matrix_size = hungarian_init(&track_det_pr, m_track_det , distTrDetmat.rows,distTrDetmat.cols, HUNGARIAN_MODE_MINIMIZE_COST) ;
    hungarian_solve(&track_det_pr);

    freeIntMat(m_track_det,distTrDetmat.rows);
    m_track_det = hungarian_get_assignment(&track_det_pr);
    distTrDetmat = intArrayToMat(m_track_det, distTrDetmat.rows, distTrDetmat.cols);

    hungarian_free(&track_det_pr);//construct distance matrix
}

//-------------- EVALUATION ----------------//

// This function will count number of files in the directory
int readDir(void){
//! Counting number in directory
/*!
Directory name is "tracker_data_files"
*/
	DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (tracker_data_files.c_str())) != NULL) {
        /* print all the files and directories within directory */
        // count file
        int count_file=0;
        while ((ent = readdir (dir)) != NULL) {
            if (ent->d_type == DT_REG) {
                count_file++;
            }
        }
        num_files = count_file;
        closedir (dir);
        return EXIT_SUCCESS;
    } else {
        /* could not open directory */
    	cerr << "Could not open directory" << std::endl;
        return EXIT_FAILURE;
    }
}



void readFile(Mat& groundtruth_mat, Mat& tracker_mat, int num_file){
//! Read tracker data and ground truth data
/*!
Output is stored in groundtruth_mat for groundtruth data and tracker_mat for tracker data
*/
	string line;

    sprintf(filename, "%sTracker_%04d.txt",tracker_data_files.c_str(),num_file);
    ifstream trfile;

    trfile.open(filename, ifstream::in);
    if (trfile.is_open()) {
         Mat new_row(1,5,CV_32S);
        while (getline(trfile, line)) {
            stringstream stream(line);
            int n;
            int it = 0;
            while(stream >> n){
                switch (it) {
                    case 0:
                        new_row.at<int>(0,0) = n;
                        break;
                    case 1:
                        new_row.at<int>(0,1) = n;
                        break;
                    case 2:
                        new_row.at<int>(0,2) = n;
                        break;
                    case 3:
                        new_row.at<int>(0,3) = n;
                        break;
                    case 4:
                        new_row.at<int>(0,4) = n;
                        break;
                    default:
                        break;
                }
                it++;
            }
            tracker_mat.push_back(new_row.row(0));
        }
        trfile.close();
    }
    else{
        cout << "File " << filename << " cannot be opened" << endl;
    }


    //groundtruth
    sprintf(filename, "%sframe_%.4d.txt",ground_truth_files.c_str(),num_file);
    ifstream gtfile;//(filename);

    gtfile.open(filename, ifstream::in);
    if (gtfile.is_open()) {
        Mat new_row(1,5,CV_32S);
        while (getline(gtfile, line)) {
            stringstream stream(line);
            int n;
            int it = 0;
            while(stream >> n){
                switch (it) {
                    case 1:
                        new_row.at<int>(0,0) = n;
                        break;
                    case 2:
                        new_row.at<int>(0,1) = n;
                        break;
                    case 3:
                        new_row.at<int>(0,2) = n;
                        break;
                    case 4:
                        new_row.at<int>(0,3) = n;
                        break;
                    case 5:
                        new_row.at<int>(0,4) = n;
                        break;
                    default:
                        break;
                }
                it++;
            }
            groundtruth_mat.push_back(new_row.row(0));
        }
        gtfile.close();
    }
    else{
        cout << "File " << filename << " cannot be opened" << endl;
    }
}


void distanceGTTracker(int num_gt,int num_tracker, Mat_<int>& distGTTr, Mat& groundtruth_mat, Mat& tracker_mat ){
//! Calculate distance between ground truth and trackers
/*!
 Output is stored in distGTTr.
 Ground truth as row
 Trackers as column
*/
	//construct distance between detection and previous detection matrix
    distGTTr = Mat_<int>::zeros(num_gt,num_tracker);
    for (int i = 0; i < num_gt; i++) {
        for (int j = 0; j < num_tracker; j++) {
            Point gtpoint;
            Point trpoint;
            gtpoint.x = groundtruth_mat.at<int>(i,0) + (int)(groundtruth_mat.at<int>(i,2)/2);
            gtpoint.y = groundtruth_mat.at<int>(i,1) + (int)(groundtruth_mat.at<int>(i,3)/2);
            trpoint.x = tracker_mat.at<int>(j,0) + (int)(tracker_mat.at<int>(j,2)/2);
            trpoint.y = tracker_mat.at<int>(j,1) + (int)(tracker_mat.at<int>(j,3)/2);
            double dist = norm(gtpoint - trpoint);
            distGTTr.at<int>(i,j) = (int)dist;
        }
    }

    hungarian_problem_t gt_tr_pr;
    int** m_gt_tr = matToIntArray(distGTTr);
    int matrix_size = hungarian_init(&gt_tr_pr, m_gt_tr , distGTTr.rows,distGTTr.cols, HUNGARIAN_MODE_MINIMIZE_COST) ;
    hungarian_solve(&gt_tr_pr);
    freeIntMat(m_gt_tr,distGTTr.rows);

    m_gt_tr = hungarian_get_assignment(&gt_tr_pr);
    distGTTr = intArrayToMat(m_gt_tr, distGTTr.rows, distGTTr.cols);
    hungarian_free(&gt_tr_pr);
}

void Evaluation(int iterate_num){
//! Function to evaluate the trackers
/*!
 Output is stored in "eval_result_files" directory.
*/
	ofstream FNfile;
    ofstream FPfile;
    ofstream TPfile;
    ofstream IDSWfile;
    fstream evalfile;
    char strbuf[30];
    readDir();
    sprintf(strbuf, "%sEval%d_/FN.txt", eval_result_files.c_str(),iterate_num);
    FNfile.open(strbuf);
    sprintf(strbuf, "%sEval%d_/FP.txt", eval_result_files.c_str(),iterate_num);
    FPfile.open(strbuf);
    sprintf(strbuf, "%sEval%d_/TP.txt", eval_result_files.c_str(),iterate_num);
    TPfile.open(strbuf);
    sprintf(strbuf, "%sEval%d_/IDSW.txt", eval_result_files.c_str(),iterate_num);
    IDSWfile.open(strbuf);
    if(!FNfile || !FPfile || !TPfile || !IDSWfile){
    	cout<<"OPEN FN FP TP IDSW file error"<<endl;
    }
    else
    {
		double MOTA = 0.0;
		double MOTP = 0.0;
		int FNtotal = 0;
		int FPtotal = 0;
		int TPtotal = 0;
		int IDSWtotal = 0;
		int FN = 0;
		int FP = 0;
		int TP = 0;
		int IDSW = 0;
		int GT = 0;
		int detMOTP = 0;
		Mat tracker_mat_prev;
		for (int nf = 0; nf < num_files; nf++) {
			Mat_<int> distGTTr;
			Mat groundtruth_mat;
			Mat tracker_mat;

			FN = 0;
			FP = 0;
			TP = 0;
			IDSW = 0;
			readFile(groundtruth_mat, tracker_mat, nf);

			int num_gt = groundtruth_mat.rows;
			int num_tr = tracker_mat.rows;

			if (num_tr < num_gt) {
				FN = groundtruth_mat.rows - tracker_mat.rows;
				FNtotal += FN;
			}
			GT += num_gt;
			distanceGTTracker(num_gt,num_tr, distGTTr, groundtruth_mat, tracker_mat);

			bool nomatch = true;
			vector<Point_<int> > pair_match;
			vector<int> nomatch_idx;
			for (int i = 0; i < distGTTr.rows; i++) {
				nomatch = true;
				for (int j = 0; j < distGTTr.cols; j++) {
					if (distGTTr[i][j]==1) {
						Point p = Point(i,j);
						pair_match.push_back(p);
						nomatch = false;
						break;
					}
				}
				if (nomatch) {
					nomatch_idx.push_back(i);
				}
			}
			for (unsigned int i = 0; i < pair_match.size(); i++) {
				Rect rgt = Rect(groundtruth_mat.at<int>(pair_match[i].x,0),groundtruth_mat.at<int>(pair_match[i].x,1),groundtruth_mat.at<int>(pair_match[i].x,2),groundtruth_mat.at<int>(pair_match[i].x,3));
				Rect rtr = Rect(tracker_mat.at<int>(pair_match[i].y,0),tracker_mat.at<int>(pair_match[i].y,1),tracker_mat.at<int>(pair_match[i].y,2),tracker_mat.at<int>(pair_match[i].y,3));
				Rect intersect_box = rgt & rtr;
				Rect union_box = rgt | rtr;
				double diff = double(intersect_box.area()) / double(union_box.area());
				if (diff > eval_intersect_threshold) {
					TP++;
				}
				else{
					FP++;
				}
				if (diff > MOTP_threshold) {
					detMOTP++;
				}
			}
			if (nf > 1) {
				pair_match.clear();
				int num_tr_prev = tracker_mat_prev.rows;
				Mat_<int> distPrevTr;
				distanceGTTracker(num_tr_prev,num_tr, distPrevTr, tracker_mat_prev, tracker_mat);
				for (int i = 0; i < distPrevTr.rows; i++) {
					nomatch = true;
					for (int j = 0; j < distPrevTr.cols; j++) {
						if (distPrevTr[i][j]==1) {
							Point p = Point(i,j);
							pair_match.push_back(p);
							nomatch = false;
							break;
						}
					}
				}
				for (unsigned int i = 0; i < pair_match.size(); i++) {
					if (tracker_mat_prev.at<int>(pair_match[i].x,4) != tracker_mat.at<int>(pair_match[i].y,4)) {
						IDSW++;
					}
				}

			}
			TPtotal += TP;
			FPtotal += FP;
			IDSWtotal += IDSW;

			sprintf(strbuf, "%d %d\n",nf, FN);
			FNfile << strbuf;
			sprintf(strbuf, "%d %d\n",nf, FP);
			FPfile << strbuf;
			sprintf(strbuf, "%d %d\n",nf, TP);
			TPfile << strbuf;
			sprintf(strbuf, "%d %d\n",nf, IDSW);
			IDSWfile << strbuf;
			tracker_mat_prev = tracker_mat;

		}
		FNfile.close();
		FPfile.close();
		TPfile.close();
		IDSWfile.close();
		double var= (double)(FNtotal + FPtotal + IDSWtotal)/(double)(GT);
		MOTA = 1.0 - (var);
		MOTP = (double)(detMOTP) / (double)(TPtotal);
		sprintf(strbuf, "%sEval%d_/Evalfile.txt",eval_result_files.c_str(),iterate_num);
		evalfile.open (strbuf,  ofstream::out | ofstream::app);
		// This part will write the data to evaluate directory with the first data is sigma_propagate
		// the first data could be changed as an x-axis when we want to plot the data
		evalfile << sigma_propagate_[0]<<" "<<FNtotal<<" "<<FPtotal<<" "<<TPtotal<<" "<<IDSWtotal<<" "<<MOTA<<" "<<MOTP<<"\n";
		evalfile.close();
    }
}

int main(int argc, const char * argv[]) {

    //initialize
    ofstream evalfile;
    read_configuration_main();
    int num_exp_local = num_experiment;

    int vis_type_local = vis_type;

    cout <<  tracker_data_files << endl;
    cout << ground_truth_files << endl;
    cout << eval_result_files << endl;
    cout << sequence_frame_files << endl;
    cout << result_tracker_files << endl;
    for (int it = 0; it < 1; it++) { // Number of iteration
        sprintf(filename, "%sEval%d_/Evalfile.txt", eval_result_files.c_str(),it);
        cout<<filename<< endl;
        evalfile.open (filename);
        evalfile.close();

        for(int expnum = 0; expnum < num_exp_local; expnum++){
            read_configuration(expnum);

            Mat detection_mat_prev, detection_mat, occlusion_mat;

            int index_of_detection = 0;

            Mat frameMat, framePrevMat;
            run_detection(0, detection_mat_prev);
            vector<ParticleFilter> Tracker(detection_mat.rows);
            // number of detections
            int num_detection;
            int num_detection_prev = detection_mat_prev.rows;

            getFrame(0, frameMat);
           //initiate particles
            for (int i = 0; i < num_detection_prev; i++) {

                index_of_detection++;
                detection_mat_prev.at<int>(i,4) = index_of_detection;
                Rect detectionBB;
                detectionBB.x = detection_mat_prev.at<int>(i,0);
                detectionBB.y = detection_mat_prev.at<int>(i,1);
                detectionBB.width = detection_mat_prev.at<int>(i,2);
                detectionBB.height = detection_mat_prev.at<int>(i,3);
                ParticleFilter newtracker(frameMat,number_of_particles,index_of_detection,0, detectionBB, use_histogram, frameMat.cols, frameMat.rows);
                newtracker.getMainData(sigma_propagate_, sigma_measurement_, c_det_, c_col_, m_c_col_, m_c_det_, _width_, _height_);
                Tracker.push_back(newtracker);
            }

            int boundary_th = 100;
            const int maxframe = 794;
            const int processed_frame = 200;
            Mat_<int> distTrDetmat; // distance betwen tracker and current detection
            Mat_<int> distDetPrev;  // distance between current and previous detection
            Mat_<int> distOccDet;
            vector<int> occlusion_idx;
            vector<Rect> occlusion_rect;
            vector<int> detection_index;
            Mat psuedo_detmat;
            LogTracker(&Tracker, 0);
            for (int frame = 1; frame < processed_frame; frame++) {
                cout << "frame " << frame << endl;

                getFrame(frame, frameMat);
                getFrame(frame-1, framePrevMat);

                //get new detection
                run_detection(frame, detection_mat);

                num_detection = detection_mat.rows;
                num_detection_prev = detection_mat_prev.rows;


                //construct distance between detection and previous detection matrix
                distancePrevAndDet(num_detection_prev, num_detection, distDetPrev, detection_mat,detection_mat_prev);

                //construct distance matrix tracker and current detection
                distanceTrackerAndDet(Tracker,num_detection, distTrDetmat, detection_mat);

                //index maintenance
                vector<Point_<int> > pair_transform;
                vector<Point_<int> > pair_occlusion;
                vector<int> nomatch_idx;
                vector<int> noise_idx;


                if (num_detection >= num_detection_prev) {
                    bool nomatch = true;
                    for (int i = 0; i < distDetPrev.cols; i++) {
                        nomatch = true;
                        for (int j = 0; j < distDetPrev.rows; j++) {
                            if (distDetPrev[j][i]==1) {
                                Point p = Point(j,i);
                                pair_transform.push_back(p);
                                nomatch = false;
                                break;
                            }
                        }
                        if (nomatch) {
                            nomatch_idx.push_back(i);
                        }
                    }

                    for (unsigned int i = 0; i < pair_transform.size(); i++) {
                        detection_mat.at<int>(pair_transform[i].y,4) = detection_mat_prev.at<int>(pair_transform[i].x,4);
                    }

                    vector<Point> pair_split; // join_split
                    if(num_detection > num_detection_prev){
                        for (unsigned int nm = 0; nm < nomatch_idx.size(); nm++) {
                            double shortest_dist = 1000.0;
                            int shortest_id = 0;
                            for (int iddet = 0; iddet < detection_mat.rows; iddet++) {
                                if (!(nomatch_idx[nm] == iddet)) {
                                    double dist = norm(Point(detection_mat.at<int>(nomatch_idx[nm],X_POS),detection_mat.at<int>(nomatch_idx[nm],Y_POS)) - Point(detection_mat.at<int>(iddet,X_POS), detection_mat.at<int>(iddet,Y_POS)));
                                    if (dist  < shortest_dist) {
                                        shortest_dist = dist;
                                        shortest_id = iddet;
                                    }
                                }

                            }

                            if (shortest_dist < 50.0) {
                                Point sp(nomatch_idx[nm],detection_mat.at<int>(shortest_id,4));
                                pair_split.push_back(sp);
                            }
                        }
                    }
                    if(with_occlusion_handling==1){
						if(occlusion_mat.rows > 0){ // join_split
							//process the pair
							for (unsigned int idxsplit = 0; idxsplit < pair_split.size(); idxsplit++) {
								for (int idxocc = 0; idxocc < occlusion_mat.rows; idxocc++) {
									if (pair_split[idxsplit].y == occlusion_mat.at<int>(idxocc,5)) {
										detection_mat.at<int>(pair_split[idxsplit].x,4) = occlusion_mat.at<int>(idxocc,4);
										Tracker[occlusion_mat.at<int>(idxocc,4)].setRelyDetection(true);
										Tracker[occlusion_mat.at<int>(idxocc,4)].resetTimesOccluded();

										for (unsigned int nm = 0; nm < nomatch_idx.size(); nm++) {
											if (nomatch_idx[nm] == pair_split[idxsplit].x) {
												for (unsigned int j = nm; j < nomatch_idx.size() - 1; j++) {
													nomatch_idx[nm] = nomatch_idx[nm+1];
												}
												nomatch_idx.resize(nomatch_idx.size()-1);
												break;
											}
										}

										for (int j = idxocc; j < occlusion_mat.rows - 1; j++) {
											occlusion_mat.row(j+1).copyTo(occlusion_mat.row(j));
										}
										Mat octemp;
										octemp = occlusion_mat.rowRange(0, occlusion_mat.rows-1);
										occlusion_mat = octemp;
									}
								}
							}
						}
                    }
                    for (unsigned int nm = 0; nm < nomatch_idx.size(); nm++) {
                        int xnew = detection_mat.at<int>(nomatch_idx[nm],0);
                        int hwidth = (int)(detection_mat.at<int>(nomatch_idx[nm],2)/2);
                        int ynew = detection_mat.at<int>(nomatch_idx[nm],1);
                        int hheight = (int)(detection_mat.at<int>(nomatch_idx[nm],3)/2);
                        Point point_detection = Point_<int>(xnew+hwidth,ynew+hheight);
                        if (isInBoundary(point_detection,frameMat,boundary_th)) {
                            //create new particle
                            Rect detectionBB;
                            index_of_detection++;
                            detection_mat.at<int>(nomatch_idx[nm],4) = index_of_detection;
                            detectionBB.x = detection_mat.at<int>(nomatch_idx[nm],0);
                            detectionBB.y = detection_mat.at<int>(nomatch_idx[nm],1);
                            detectionBB.width = detection_mat.at<int>(nomatch_idx[nm],2);
                            detectionBB.height = detection_mat.at<int>(nomatch_idx[nm],3);
                            ParticleFilter newTracker(frameMat, number_of_particles,detection_mat.at<int>(nomatch_idx[nm],4),frame,detectionBB, use_histogram, frameMat.cols, frameMat.rows);
                            newTracker.getMainData(sigma_propagate_, sigma_measurement_, c_det_, c_col_, m_c_col_, m_c_det_, _width_, _height_);
                            Tracker.push_back(newTracker);
                        }
                        else
                        {	if(with_occlusion_handling==1){
								if(occlusion_mat.rows > 0){
									pair_occlusion.clear();
									distanceOcclusionDet(occlusion_mat.rows, detection_mat.rows, distOccDet, detection_mat, nomatch_idx[nm], occlusion_mat);
									for (int i = 0; i < distOccDet.rows; i++) {
										nomatch = true;
										for (int j = 0; j < distOccDet.cols; j++) {
											if (distOccDet[j][i]==1) {
												Point p = Point(i,nomatch_idx[nm]);
												pair_occlusion.push_back(p);
												nomatch = false;
												break;
											}
										}
									}

									bool matchtracker = false;
									int index_tracker;
									for (unsigned int i = 0; i < pair_occlusion.size(); i++) {
										matchtracker = false;
										if (occlusion_mat.at<int>(pair_occlusion[i].x,5)==0) { // join_split
											int xnew = occlusion_mat.at<int>(pair_occlusion[i].x,0);
											int hwidth = (int)(occlusion_mat.at<int>(pair_occlusion[i].x,2)/2);
											int ynew = occlusion_mat.at<int>(pair_occlusion[i].x,1);
											int hheight = (int)(occlusion_mat.at<int>(pair_occlusion[i].x,3)/2);
											Point pointnew = Point_<int>(xnew+hwidth,ynew+hheight);
											if (isInBoundary(pointnew, frameMat, boundary_th)) {

											}
											else
											{

												int indexfordet = (int)(occlusion_mat.at<int>(pair_occlusion[i].x,4));
												detection_mat.at<int>(pair_occlusion[i].y,4) = indexfordet;

												for (unsigned int t = 0; t < Tracker.size(); t++) {
													if (Tracker[t].getDetectionIndex() == occlusion_mat.at<int>(pair_occlusion[i].x,4)) {
														matchtracker = true;
														index_tracker = t;
														break;
													}
												}
												if (matchtracker) {
													Tracker[index_tracker].setRelyDetection(true);
													Tracker[index_tracker].resetTimesOccluded();
													for (int j = pair_occlusion[i].x; j < occlusion_mat.rows - 1; j++) {
														occlusion_mat.row(j+1).copyTo(occlusion_mat.row(j));
													}
													Mat octemp;
													octemp = occlusion_mat.rowRange(0, occlusion_mat.rows-1);
													occlusion_mat = octemp;
												}
											}
										}
									}
								}
                        	}
                            occlusion_idx.clear();
                        }
                    }
                    pair_occlusion.clear();
                    nomatch_idx.clear();
                }
                if (num_detection < num_detection_prev) {
                     if (Tracker.size() >= num_detection){
                        bool nomatch = true;
                        for (int i = 0; i < distDetPrev.rows; i++) {
                            nomatch = true;
                            for (int j = 0; j < distDetPrev.cols; j++) {
                                if (distDetPrev[i][j]==1) {
                                    Point p = Point(i,j);
                                    pair_transform.push_back(p);
                                    nomatch = false;
                                    break;
                                }
                            }
                            if (nomatch) {
                                nomatch_idx.push_back(i);
                            }
                        }
                        for (unsigned int i = 0; i < pair_transform.size(); i++) {
                            detection_mat.at<int>(pair_transform[i].y,4) = detection_mat_prev.at<int>(pair_transform[i].x,4);
                        }
                        for (unsigned int i = 0; i < nomatch_idx.size(); i++) {
                            Mat new_row = Mat::zeros(1,6,CV_32S);
                            int x_ = detection_mat_prev.at<int>(nomatch_idx[i],0);
                            int y_ = detection_mat_prev.at<int>(nomatch_idx[i],1);
                            int width_ = detection_mat_prev.at<int>(nomatch_idx[i],2);
                            int height_ = detection_mat_prev.at<int>(nomatch_idx[i],3);
                            Point pdet;
                            pdet.x = x_ + (width_/2);
                            pdet.y = y_ + (height_/2);
                            if (isInBoundary(pdet,frameMat,boundary_th)) {

                            }
                            else{
                                detection_mat_prev.row(nomatch_idx[i]).copyTo(new_row.row(0).colRange(0, NUM_STATES+1));
                                double shortest_dist = 1000.0; // join_split
                                int shortest_id = 0;
                                for (int iddetprev = 0; iddetprev < detection_mat_prev.rows; iddetprev++) {
                                    if (!(nomatch_idx[i] == iddetprev)) {
                                        double dist = norm(Point(detection_mat_prev.at<int>(nomatch_idx[i],X_POS),detection_mat_prev.at<int>(nomatch_idx[i],Y_POS)) - Point(detection_mat_prev.at<int>(iddetprev,X_POS), detection_mat_prev.at<int>(iddetprev,Y_POS)));
                                        //                                    cout  << detection_mat_prev.at<int>(nomatch_idx[i],4) << " " << detection_mat_prev.at<int>(iddetprev,4) << " "<<dist << endl;
                                        if (dist  < shortest_dist) {
                                            shortest_dist = dist;
                                            shortest_id = iddetprev;
                                        }
                                    }

                                }

                                if (shortest_dist < 20.0) {
                                    new_row.at<int>(0,5) = detection_mat_prev.at<int>(shortest_id,4);
                                }
                                else{
                                    new_row.at<int>(0,5) = 0;
                                }
                                if(with_occlusion_handling==1){
									occlusion_mat.push_back(new_row.row(0));
                               }
							}
                        }
                        vector<int> trackernomatch_idx;
                        //             indexing match
                        for (int i = 0; i < distTrDetmat.rows; i++) {
                            nomatch = true;
                            for (int j = 0; j < distTrDetmat.cols; j++) {
                                if (distTrDetmat[i][j]==1) {
                                    nomatch = false;
                                    break;
                                }
                            }
                            if (nomatch) {
                                Point point_tracker = Tracker[i].getMeanPosition();

								if (isInBoundary(point_tracker,frameMat,boundary_th)) {
	                                trackernomatch_idx.push_back(i);
								}
                            }
                        }
                        if(with_occlusion_handling==1){
							if (occlusion_mat.rows > 0) {
								bool match = false;
								int tracker_idx;

								for (int i = 0; i < occlusion_mat.rows; i++) {
									match = false;
									for (unsigned int trac = 0; trac < Tracker.size(); trac++) {
										if (Tracker[trac].getDetectionIndex()==occlusion_mat.at<int>(i,4)) {
											tracker_idx = trac;
											match = true;
											break;
										}
									}
									if (match) {
										Tracker[tracker_idx].setRelyDetection(false);
										Tracker[tracker_idx].incTimesOccluded();
									}
								}
							}
                        }
                        // Tracker is in boundary = object leaving scene
                        for (unsigned int tr = 0; tr < trackernomatch_idx.size(); tr++) {
                           Tracker.erase(Tracker.begin() + trackernomatch_idx[tr]);

                        }

                        trackernomatch_idx.clear();
                    }
                }
                if (num_detection == num_detection_prev) {
                    bool nomatch = true;
                    for (int i = 0; i < distDetPrev.rows; i++) {
                        nomatch = true;
                        for (int j = 0; j < distDetPrev.cols; j++) {
                            if (distDetPrev[i][j]==1) {
                                Point p = Point(i,j);
                                pair_transform.push_back(p);
                                nomatch = false;
                                break;
                            }
                        }
                        if (nomatch) {
                            nomatch_idx.push_back(i);
                        }
                    }

                    for (unsigned int i = 0; i < pair_transform.size(); i++) {
                        detection_mat.at<int>(pair_transform[i].y,4) = detection_mat_prev.at<int>(pair_transform[i].x,4);
                    }

                    //modify occlusion mat
                    if(with_occlusion_handling==1){
						if (occlusion_mat.rows>0) {
							bool match = false;
							int tracker_index;
							for (int i = 0; i < occlusion_mat.rows; i++) {
								for (unsigned int trac = 0; trac < Tracker.size(); trac++) {
									if (Tracker[trac].getDetectionIndex()==occlusion_mat.at<int>(i,4)) {
										tracker_index = trac;
										match = true;
										break;
									}
								}
								if (match) {
									Tracker[tracker_index].incTimesOccluded();
									if (Tracker[tracker_index].getTimesOccluded() > 2) {
										int xnew = Tracker[tracker_index].getMeanPosition().x;// - (occlusion_mat.at<int>(i,2)/2);
										int ynew = Tracker[tracker_index].getMeanPosition().y;// - (occlusion_mat.at<int>(i,3)/2);
										occlusion_mat.at<int>(i,0) = xnew;
										occlusion_mat.at<int>(i,1) = ynew;
									}
									else
									{
										int xnew = Tracker[tracker_index].getMeanPosition().x - (occlusion_mat.at<int>(i,2)/2);
										int ynew = Tracker[tracker_index].getMeanPosition().y - (occlusion_mat.at<int>(i,3)/2);
										occlusion_mat.at<int>(i,0) = xnew;
										occlusion_mat.at<int>(i,1) = ynew;
									}

								}
							}
						}

					}
                }

                vector<Point>().swap(pair_transform);
                vector<int>().swap(nomatch_idx);
                distanceTrackerAndDet(Tracker,num_detection, distTrDetmat, detection_mat);
                if(with_occlusion_handling==1){
					if (occlusion_mat.rows > 0) {
						bool match = false;
						int index_tr;

						match = false;

						for (int i = 0; i < occlusion_mat.rows; i++) {
							for (unsigned int j = 0; j < Tracker.size(); j++) {
								if (Tracker[j].getRelyDetection()== false && Tracker[j].getDetectionIndex()==occlusion_mat.at<int>(i,4)) {
									match = true;
									index_tr = j;
									break;
								}
							}
							if (match) {
								Rect BB;
								int width = occlusion_mat.at<int>(i,2);
								int height =  occlusion_mat.at<int>(i,3);
								BB.x = Tracker[index_tr].getMeanPosition().x - width/2;
								BB.y = Tracker[index_tr].getMeanPosition().y - height/2;
								BB.width = width;
								BB.height = height;
								Tracker[index_tr].prediction(frameMat);
								bool success = Tracker[index_tr].update(frameMat, framePrevMat, BB, use_histogram, frame);
								writeNewImage(frameMat,frame,vis_type_local, Tracker[index_tr], BB, Tracker[index_tr].getDetectionIndex());
							}
						}
					}

                //predict
                Rect detectionBB;
                bool match = false;
                int index_tr;

                cout << "Trackers "<<endl;
                    for (unsigned int i = 0; i < Tracker.size(); i++) {
                        match = false;
                        if (Tracker[i].getRelyDetection()!=false) {
                            for (int j = 0; j < detection_mat.rows; j++) {
                                if (detection_mat.at<int>(j,4)==Tracker[i].getDetectionIndex()) {
                                    match = true;
                                    index_tr = j;
                                    break;
                                }
                            }
                            if (match) {
                                Rect BB;
                                int width = detection_mat.at<int>(index_tr,2);
                                int height =  detection_mat.at<int>(index_tr,3);
                                BB.x = detection_mat.at<int>(index_tr,0);
                                BB.y = detection_mat.at<int>(index_tr,1);
                                BB.width = width;
                                BB.height = height;
                                Tracker[i].prediction(frameMat);
                                bool success = Tracker[i].update(frameMat, framePrevMat, BB, use_histogram, frame);
                                writeNewImage(frameMat,frame,vis_type_local, Tracker[i], BB, Tracker[i].getDetectionIndex());
                            }
                            else{

                            }
                        }
                    }
                }
                else{
                    vector<int> trackermatch_idx;
                    //             indexing match
                    bool match;
                    int j_det;
                    for (int i = 0; i < distTrDetmat.rows; i++) {
                        match = false;
                        for (int j = 0; j < distTrDetmat.cols; j++) {
                            if (distTrDetmat[i][j]==1) {
                                match = true;
                                j_det = j;
                                break;
                            }
                        }
                        if (match) {
                            Rect BB;
                            int width = detection_mat.at<int>(j_det,2);
                            int height =  detection_mat.at<int>(j_det,3);
                            BB.x = detection_mat.at<int>(j_det,0);
                            BB.y = detection_mat.at<int>(j_det,1);
                            BB.width = width;
                            BB.height = height;
                            Tracker[i].prediction(frameMat);
                            bool success = Tracker[i].update(frameMat, framePrevMat, BB, use_histogram, frame);
                            writeNewImage(frameMat,frame,vis_type_local, Tracker[i], BB, Tracker[i].getDetectionIndex());

                        }
                    }

                }
                LogTracker(&Tracker, frame);
                detection_mat.copyTo(detection_mat_prev);
                pair_transform.clear();
                pair_occlusion.clear();
            }

            Evaluation(it);
            Tracker.clear();

            cout << "done" << endl;
        }
        cout << "finish" << endl;

    }
    cout << "END" << endl;
    return 0;
}

