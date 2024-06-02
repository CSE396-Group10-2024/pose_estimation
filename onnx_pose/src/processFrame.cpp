#include "processFrame.h"

#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <iostream>
#include <typeinfo>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fstream>
#include <filesystem> // For std::filesystem
/* for OpenCV */
#include <opencv2/opencv.hpp>
#include "MPPose.h"
#include "MPPersonDet.h"
using namespace std;
using namespace dnn;
using namespace cv;




vector< pair<dnn::Backend, dnn::Target> > backendTargetPairs = {
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_OPENCV, dnn::DNN_TARGET_CPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA_FP16),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_TIMVX, dnn::DNN_TARGET_NPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CANN, dnn::DNN_TARGET_NPU) };


processFrame::processFrame(string person_detection_path,string pose_detection_path)
    : 
      confThreshold(0.2f),
      scoreThreshold(0.5f),
      nmsThreshold(0.3f),
      topK(5000),
      backendTargetid(3),
      personDetector(person_detection_path, nmsThreshold, scoreThreshold, topK, backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second),
      poseDetector(pose_detection_path, confThreshold, backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second) {
    // Additional initialization if needed
}
struct AngleIndexes
{
    std::string name;
 int first;
 int second;
 int third;
 int result;
 int loc_result;
};
struct DistanceIndexes{
    std::string name;
    int first;
    int second;
    int result;
    int loc_result;
    
};

std::vector<int>indeces={8,7,12,11,14,13,16,15,24,23,25,26,27,28}
;
std::vector<DistanceIndexes> pointIndexofDistances{
    {"right_arm",16,14,0,0},
    {"right_arm2",14,12,0,0},
    {"left_arm",15,13,0,0},
    {"left_arm2",13,11,0,0},
    {"right_leg",24,26,0,0},
    {"right_leg2",26,28,0,0},
    {"left_leg",23,25,0,0},
    {"left_leg2",25,27,0,0},
    {"right_body",12,24,0,0},
    {"left_body",11,23,0,0},
    {"left_head",7,11,0,0},
    {"right_head",8,12,0,0},
  
    {"left_arm_left_foot",15,27,0,0},

    {"right_arm_right_foot",16,28,0,0},


    {"right_shoulder_right_foot",12,28,0,0},
    
    {"left_shoulder_left_foot",11,27,0,0},

    {"left_foot_right_foot",27,28,0,0},

    {"left_knee_right_knee",26,25,0,0},
    {"left_hand_right_hand",16,15,0,0},

    {"right_hand_right_head",16,8,0,0},
    {"left_hand_left_head",15,7,0,0},
    

};
 std::vector<AngleIndexes>pointIndexesofAngles{
    {"right_arm_angle",16,14,12,0,0},
    {"left_arm_angle",15,13,11,0,0},
    {"right_leg_angle",28,26,24,0,0},
    {"left_leg_angle",27,25,23,0,0},
    {"right_rip_angle",12,24,26,0,0},
    {"left_rip_angle",11,23,25,0,0},
    {"right_shoulder_angle",14,12,24,0,0},
    {"left_shoulder_angle",13,11,23,0,0},
    {"right_head_angle",14,12,8,0,0},
    {"left_head_angle",13,11,7,0,0},
    {"left_head_body_angle",7,11,23,0,0},
    {"right_head_body_angle",8,12,24,0,0},

    {"right_hand_right_rip_angle",16,12,24,0,0},
    {"left_hand_left_rip_angle",15,11,23,0,0},
    {"right_shoulder_right_foot_angle",12,24,28,0,0},
    {"left_shoulder_left_foot_angle",11,23,27,0,0},
    {"left_head_left_hand_angle",7,11,15,0,0},
    {"right_head_right_hand_angle",8,12,16,0,0}
   
};




double calculate_distance(const pair<double, double>& point1, const pair<double, double>& point2) {
  double dx = point1.first - point2.first;
  double dy = point1.second - point2.second;
  return sqrt(dx * dx + dy * dy);
}

// // Function to calculate angle between two lines
// double angle_between_lines(const std::pair<double, double>& point1, const std::pair<double, double>& point2, const std::pair<double, double>& point3) {
//     std::cout << "in function " << std::endl;
//     std::cout << point1.first << " " << point1.second << std::endl;
//     std::cout << point2.first << " " << point2.second << std::endl;
//     std::cout << point3.first << " " << point3.second << std::endl;

//     double horizontal_i_j = std::abs(point2.first - point3.first);
//     double vertical_i_j = std::abs(point2.second - point3.second);
//     double dist_i_j = calculate_distance(point2, point3);
//     double p = (vertical_i_j * vertical_i_j + dist_i_j * dist_i_j - horizontal_i_j * horizontal_i_j) / (2 * vertical_i_j * dist_i_j);
//     double angle = acos(p) * 180.0 / M_PI; // Convert to degrees

//     std::cout << "Angle: " << angle << std::endl;
//     return angle;
// }

pair<double, double> normalize_keypoint(const pair<double, double>& keypoint, double img_width, double img_height, double xmin, double xmax, double ymin, double ymax) {
    double x_normalized = (img_width / 2) * (keypoint.first - xmin) / (xmax - xmin);
    double y_normalized = (img_height / 2) * (keypoint.second - ymin) / (ymax - ymin);
    return {x_normalized, y_normalized};
}
double angle_between_lines(const pair<double, double>& point1, const pair<double, double>& point2,
                             const pair<double, double>& point3, const pair<double, double>& point4) {
  // Direction vectors
  pair<double, double> direction1 = {point2.first - point1.first, point2.second - point1.second};
  pair<double, double> direction2 = {point4.first - point3.first, point4.second - point3.second};

  // Line lengths
  double line_length1 = sqrt(direction1.first * direction1.first + direction1.second * direction1.second);
  double line_length2 = sqrt(direction2.first * direction2.first + direction2.second * direction2.second);

  // Dot product
  double dot_product = direction1.first * direction2.first + direction1.second * direction2.second;

  // Handle potential zero division (collinear lines)
  if (line_length1 == 0 || line_length2 == 0) {
    return 0;
  }

  // Normalized value and angle in radians
  double normalized_value = dot_product / (line_length1 * line_length2);
  double angle_rad = acos(normalized_value);

  // Angle in degrees and absolute value
  double angle_deg = angle_rad * (180 / M_PI);
  return abs(angle_deg);
}

 void processFrame:: set_engine(const string& txt_file_name_for_data,const string& csv_file_name_for_data,bool connect_to_server,const string &mode){

    std::filesystem::path currentPath = std::filesystem::current_path();

    // Define the file name
    

    // Concatenate the current path with the file name
    std::filesystem::path txt_fullPath = currentPath / txt_file_name_for_data;
    std::filesystem::path csv_fullPath = currentPath / csv_file_name_for_data;
    this->txt_file_name_for_data = txt_fullPath;
    this->csv_file_name_for_data = csv_fullPath;
    this->mode = mode;
    this->index=0;
    if(connect_to_server){
        connect_server();
    }
     
   }




static constexpr float kThresholdScoreKeyPoint = 0.2f;
void processFrame::set_index(int index){
    this->index = index;
}
int processFrame::get_index(){
    return this->index;
}



void drawLines(Mat image, Mat landmarks, Mat keeplandmarks, bool isDrawPoint = true, int thickness = 2)
{
    
    vector<pair<int, int>> segment = {
        make_pair(0, 1), make_pair(1, 2), make_pair(2, 3), make_pair(3, 7),
        make_pair(0, 4), make_pair(4, 5), make_pair(5, 6), make_pair(6, 8),
        make_pair(9, 10),
        make_pair(12, 14), make_pair(14, 16), make_pair(16, 22), make_pair(16, 18), make_pair(16, 20), make_pair(18, 20),
        make_pair(11, 13), make_pair(13, 15), make_pair(15, 21), make_pair(15, 19), make_pair(15, 17), make_pair(17, 19),
        make_pair(11, 12), make_pair(11, 23), make_pair(23, 24), make_pair(24, 12),
        make_pair(24, 26), make_pair(26, 28), make_pair(28, 30), make_pair(28, 32), make_pair(30, 32),
        make_pair(23, 25), make_pair(25, 27),make_pair(27, 31), make_pair(27, 29), make_pair(29, 31) };
    for (auto p : segment)
        if (keeplandmarks.at<uchar>(p.first) && keeplandmarks.at<uchar>(p.second))
            line(image, Point(landmarks.row(p.first)), Point(landmarks.row(p.second)), Scalar(255, 255, 255), thickness);
    if (isDrawPoint)
        for (int idxRow = 0; idxRow < landmarks.rows; idxRow++)
            if (keeplandmarks.at<uchar>(idxRow))
                circle(image, Point(landmarks.row(idxRow)), thickness, Scalar(0, 0, 255), -1);
}
pair<Mat, Mat> visualize(Mat image, vector<tuple<Mat, Mat, Mat, Mat, Mat, float>> poses)
{
    Mat displayScreen = image.clone();
    Mat display3d(400, 400, CV_8UC3, Scalar::all(0));
    line(display3d, Point(200, 0), Point(200, 400), Scalar(255, 255, 255), 2);
    line(display3d, Point(0, 200), Point(400, 200), Scalar(255, 255, 255), 2);
    putText(display3d, "Main View", Point(0, 12), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    putText(display3d, "Top View", Point(200, 12), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    putText(display3d, "Left View", Point(0, 212), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    putText(display3d, "Right View", Point(200, 212), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    bool isDraw = false;  // ensure only one person is drawn

    for (auto pose : poses)
    {
        Mat bbox = get<0>(pose);
        if (!bbox.empty())
        {   
            Mat landmarksScreen = get<1>(pose);
            Mat landmarksWord = get<2>(pose);
            Mat mask;
            get<3>(pose).convertTo(mask, CV_8U);
            Mat heatmap = get<4>(pose);
            float conf = get<5>(pose);
            Mat edges;
            Canny(mask, edges, 100, 200);
            Mat kernel(2, 2, CV_8UC1, Scalar::all(1)); // expansion edge to 2 pixels
            dilate(edges, edges, kernel);
            Mat edgesBGR;
            cvtColor(edges, edgesBGR, COLOR_GRAY2BGR);
            Mat idxSelec = edges == 255;
            edgesBGR.setTo(Scalar(0, 255, 0), idxSelec);

            add(edgesBGR, displayScreen, displayScreen);
            // draw box
            Mat box;
            bbox.convertTo(box, CV_32S);

            // rectangle(displayScreen, Point(box.row(0)), Point(box.row(1)), Scalar(0, 255, 0), 2);
            putText(displayScreen, format("Conf = %4f", conf), Point(0, 35), FONT_HERSHEY_DUPLEX, 0.7,Scalar(0, 0, 255), 2);
             // Draw line between each key points
            landmarksScreen = landmarksScreen.rowRange(0, landmarksScreen.rows - 6);
            landmarksWord = landmarksWord.rowRange(0, landmarksWord.rows - 6);

            Mat keepLandmarks = landmarksScreen.col(4) > 0.8; // only show visible keypoints which presence bigger than 0.8

            Mat landmarksXY;
            landmarksScreen.colRange(0, 2).convertTo(landmarksXY, CV_32S);
            // drawLines(displayScreen, landmarksXY, keepLandmarks, false);

            // z value is relative to HIP, but we use constant to instead
            for (int idxRow = 0; idxRow < landmarksScreen.rows; idxRow++)
            {
                Mat landmark;// p in enumerate(landmarks_screen[:, 0 : 3].astype(np.int32))
                landmarksScreen.row(idxRow).convertTo(landmark, CV_32S);
                if (keepLandmarks.at<uchar>(idxRow))
                    circle(displayScreen, Point(landmark.at<int>(0), landmark.at<int>(1)), 2, Scalar(0, 0, 255), -1);
                
                
                
                    int radius = 10;  // Adjust radius as needed
                    cv::Scalar color(0, 0, 255); // Blue circles


                    // Put text label next to the circle
                    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
                    double fontScale =2.5;
                    int thickness = 1;
                    cv::Point textPt(landmark.at<int>(0) + radius + 5, landmark.at<int>(1)+ 5); // Adjust position

                    std::string text = std::to_string(idxRow);
                    cv::putText(displayScreen, text, textPt, fontFace, fontScale, color, thickness);
            }

            if (!isDraw)
            {
                isDraw = true;
                // Main view
                Mat landmarksXY = landmarksWord.colRange(0, 2).clone();
                Mat x = landmarksXY * 100 + 100;
                x.convertTo(landmarksXY, CV_32S);
                drawLines(display3d, landmarksXY, keepLandmarks, true, 2);

                // Top view
                Mat landmarksXZ;
                hconcat(landmarksWord.col(0), landmarksWord.col(2), landmarksXZ);
                landmarksXZ.col(1) = -landmarksXZ.col(1);
                x = landmarksXZ * 100;
                x.col(0) += 300;
                x.col(1) += 100;
                x.convertTo(landmarksXZ, CV_32S);
                drawLines(display3d, landmarksXZ, keepLandmarks, true, 2);

                // Left view
                Mat landmarksYZ;
                hconcat(landmarksWord.col(2), landmarksWord.col(1), landmarksYZ);
                landmarksYZ.col(0) = -landmarksYZ.col(0);
                x = landmarksYZ * 100;
                x.col(0) += 100;
                x.col(1) += 300;
                x.convertTo(landmarksYZ, CV_32S);
                drawLines(display3d, landmarksYZ, keepLandmarks, true, 2);

                // Right view
                Mat landmarksZY;
                hconcat(landmarksWord.col(2), landmarksWord.col(1), landmarksZY);
                x = landmarksZY * 100;
                x.col(0) += 300;
                x.col(1) += 300;
                x.convertTo(landmarksZY, CV_32S);
                drawLines(display3d, landmarksZY, keepLandmarks, true, 2);
            }
        }
    }
    return pair<Mat, Mat>(displayScreen, display3d);
}


int32_t processFrame::Process(cv::Mat& frame) // details of analyzing the image pose processes
{
    Mat person = this->personDetector.infer(frame);
    vector<tuple<Mat, Mat, Mat, Mat, Mat, float>> pose;
    for (int idxRow = 0; idxRow < person.rows; idxRow++)
    {
        tuple<Mat, Mat, Mat, Mat, Mat, float> re = this->poseDetector.infer(frame, person.row(idxRow));
        if (!get<0>(re).empty())
            pose.push_back(re);
    }

    double img_width = frame.cols;
    double img_height = frame.rows;
    double xmin = DBL_MAX, ymin = DBL_MAX, xmax = -DBL_MAX, ymax = -DBL_MAX;

    static const std::string kWinName = "MPPose Demo";
    if (pose.size()>0){
   tuple<Mat, Mat, Mat, Mat, Mat, float> temp=pose.at(0);
        Mat landmarksScreen = get<1>(temp);
        for (int i = 0; i < landmarksScreen.rows; ++i) {
        for (int j = 0; j < landmarksScreen.cols; ++j) {
            cout <<"landmark "<<i<<" "<<j<<" " <<landmarksScreen.at<float>(i, j) << " ";
        }
        cout << endl;
    }

    Mat localizedLandmarks = landmarksScreen.clone(); 
         for (size_t i{0};  i < indeces.size(); ++i) {
        int idx = indeces[i];
        double x = landmarksScreen.at<float>(idx, 0);
        double y = landmarksScreen.at<float>(idx, 1);
       
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
    }
    for (size_t i{0}; i < indeces.size(); ++i){
        pair<double,double>localized_point=normalize_keypoint({landmarksScreen.at<float>(indeces.at(i), 0), landmarksScreen.at<float>(indeces.at(i), 1)},img_width,img_height,xmin,xmax,ymin,ymax);
        localizedLandmarks.at<float>(indeces.at(i), 0) = localized_point.first;
        localizedLandmarks.at<float>(indeces.at(i), 1) = localized_point.second;
    }
        float conf = get<5>(temp);
        cout<<"confidence: "<<conf<<endl;
       for (size_t i{0}; i < indeces.size(); ++i) {
        cv::Point center(landmarksScreen.at<float>(indeces.at(i), 0), landmarksScreen.at<float>(indeces.at(i), 1));
        cv::Point center1(localizedLandmarks.at<float>(indeces.at(i), 0), localizedLandmarks.at<float>(indeces.at(i), 1));
        int radius = 10;  // Adjust radius as needed
        cv::Scalar color(0, 0, 255); // Blue circles
        cv::Scalar color1(0, 255, 255);
        cv::circle(frame, center, radius, color, cv::FILLED);
                cv::circle(frame, center1, radius, color1, cv::FILLED);

        // Put text label next to the circle
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 2.5;
        int thickness = 2;
        cv::Point textPt(center.x + radius + 5, center.y + 5); // Adjust position
        cv::Point textPt1(center1.x + radius + 5, center1.y + 5); // Adjust position

        std::string text = std::to_string(indeces.at(i));
        putText(frame, text, textPt, fontFace, fontScale, color, thickness);
        putText(frame, text, textPt1, fontFace, fontScale, color1, thickness);

    }
    //add normalized
     // Calculate angles and distances
    for (auto &AngleIndexes : pointIndexesofAngles) {
        if (conf > 0.2) {
      
            double angle = angle_between_lines(
                {double(landmarksScreen.at<float>(AngleIndexes.first, 0)), double(landmarksScreen.at<float>(AngleIndexes.first, 1))},
                {double(landmarksScreen.at<float>(AngleIndexes.second, 0)), double(landmarksScreen.at<float>(AngleIndexes.second, 1))},
                {double(landmarksScreen.at<float>(AngleIndexes.third, 0)), double(landmarksScreen.at<float>(AngleIndexes.third, 1))}  ,
                 {double(landmarksScreen.at<float>(AngleIndexes.second, 0)), double(landmarksScreen.at<float>(AngleIndexes.second, 1))} );
            double loc_angle=angle_between_lines(
                {double(localizedLandmarks.at<float>(AngleIndexes.first, 0)), double(localizedLandmarks.at<float>(AngleIndexes.first, 1))},
                {double(localizedLandmarks.at<float>(AngleIndexes.second, 0)), double(localizedLandmarks.at<float>(AngleIndexes.second, 1))},
                {double(localizedLandmarks.at<float>(AngleIndexes.third, 0)), double(localizedLandmarks.at<float>(AngleIndexes.third, 1))}
              ,{double(localizedLandmarks.at<float>(AngleIndexes.second, 0)), double(localizedLandmarks.at<float>(AngleIndexes.second, 1))} 
            );
            AngleIndexes.result = angle;
            AngleIndexes.loc_result=loc_angle;
        } else {
            AngleIndexes.result = 0;
            AngleIndexes.loc_result=0;
        }
    }
    for (auto &DistanceIndexes : pointIndexofDistances) {
        if (conf > 0.2) {
            double distance = calculate_distance(
                {landmarksScreen.at<float>(DistanceIndexes.first, 0), landmarksScreen.at<float>(DistanceIndexes.first, 1)},
                {landmarksScreen.at<float>(DistanceIndexes.second, 0), landmarksScreen.at<float>(DistanceIndexes.second, 1)}
            );
            double loc_distance = calculate_distance(
                {localizedLandmarks.at<float>(DistanceIndexes.first, 0), localizedLandmarks.at<float>(DistanceIndexes.first, 1)},
                {localizedLandmarks.at<float>(DistanceIndexes.second, 0), localizedLandmarks.at<float>(DistanceIndexes.second, 1)}
            );
            DistanceIndexes.loc_result=loc_distance;
            DistanceIndexes.result = distance;
        } else {
            DistanceIndexes.result = 0;
            DistanceIndexes.loc_result=0;
        }
    }

    // Create message
    char message[1024];
    message[0] = '{';
    int current_position = 1;
    for (auto &AngleIndexes : pointIndexesofAngles) {
        current_position += sprintf(message + current_position, "\"%s\":%d,", AngleIndexes.name.c_str(), AngleIndexes.result);
    }
    // for (auto &DistanceIndexes : pointIndexofDistances) {
    //     current_position += sprintf(message + current_position, "\"%s\":%d,", DistanceIndexes.name.c_str(), DistanceIndexes.result);
    // }
    message[current_position - 1] = '}';

    cout << endl;
    cout << "message to python: " << message << endl;

    // Handle message based on mode
    if (mode == "send") {
        if (send(clientSocket, message, strlen(message), 0) == -1) {
            std::cerr << "Error sending message: " << strerror(errno) << std::endl;
        }
        while(true){
        char buffer[1024];
        memset(buffer, 0, 1024);

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(clientSocket, &readfds);
        float timeout_sec=0.2;
        struct timeval timeout;
        timeout.tv_sec =timeout_sec;
        timeout.tv_usec = timeout_sec*1000000;

        int activity = select(clientSocket + 1, &readfds, NULL, NULL, &timeout);

        if (activity < 0) {
            std::cerr << "Select error: " << strerror(errno) << std::endl;
        } else if (activity == 0) {
            std::cout << "Timeout: No data received for " << 0.5 << " seconds." << std::endl;
        }
        else{
            ssize_t bytesReceived = recv(clientSocket, buffer, 1024 - 1, 0);
            if (bytesReceived < 0) {
                std::cerr << "Error receiving message: " << strerror(errno) << std::endl;
            } else if (bytesReceived == 0) {
                std::cout << "Echo can not received." << std::endl;
            }
            std::cout << "Received echo: " << buffer << std::endl;

             if(strncmp(buffer,message,1024)==0){
                break;
            }
   
   
        }
        }
 
    
    } 
    else if (mode == "database") {
        std::ofstream txt_file_data(this->txt_file_name_for_data, std::ios_base::app);
        std::ofstream csv_file_data(this->csv_file_name_for_data, std::ios_base::app);

        // Write headers for keypoints and their attributes
       

        // Write keypoint data
        
        // Write headers for angle and distance calculations
        if (index == 0) {
            csv_file_data << "index";
            for (size_t i = 0; i < indeces.size(); ++i) {
                int keypoint_index = indeces[i];
                csv_file_data << ",x" << std::to_string(keypoint_index)
                              << ",y" << std::to_string(keypoint_index)
                              << ",z" << std::to_string(keypoint_index)
                              << ",visibility" << std::to_string(keypoint_index)
                              << ",presence" << std::to_string(keypoint_index)
                             << ",x_loc" << std::to_string(keypoint_index)
                             << ",y_loc" << std::to_string(keypoint_index);

            }
            csv_file_data << ",";
            for (auto &AngleIndexes : pointIndexesofAngles) {
                csv_file_data << AngleIndexes.name << ",";
            }
            for (auto &DistanceIndexes : pointIndexofDistances) {
                csv_file_data << DistanceIndexes.name << ",";
            }
            for (auto &AngleIndexes : pointIndexesofAngles) {
                csv_file_data << "localized_"<<AngleIndexes.name << ",";
            }
            for (auto &DistanceIndexes : pointIndexofDistances) {
                csv_file_data <<"localized_" <<DistanceIndexes.name << ",";
            }
            csv_file_data << std::endl;
        }
        // Write angle and distance results
        csv_file_data << index ;
        for (size_t i{0}; i < indeces.size(); ++i) {
            int keypoint_index = indeces[i];
            csv_file_data << "," << landmarksScreen.at<float>(keypoint_index, 0)
                          << "," << landmarksScreen.at<float>(keypoint_index, 1)
                          << "," << landmarksScreen.at<float>(keypoint_index, 2)
                          << "," << landmarksScreen.at<float>(keypoint_index, 3)
                          << "," << landmarksScreen.at<float>(keypoint_index, 4)
                          << "," << localizedLandmarks.at<float>(keypoint_index, 0)
                          << "," << localizedLandmarks.at<float>(keypoint_index, 1);
        }
        csv_file_data<<",";
        for (auto &AngleIndexes : pointIndexesofAngles) {
            csv_file_data << AngleIndexes.result << ",";
        }
        for (auto &DistanceIndexes : pointIndexofDistances) {
            csv_file_data << DistanceIndexes.result << ",";
        }
        for (auto &AngleIndexes : pointIndexesofAngles) {
            csv_file_data << AngleIndexes.loc_result << ",";
        }
        for (auto &DistanceIndexes : pointIndexofDistances) {
            csv_file_data << DistanceIndexes.loc_result << ",";
        }
        csv_file_data << std::endl;
    }
    int window_width = 240;
    int window_height = 180;
      namedWindow("Display frame", WINDOW_NORMAL);
      moveWindow("win1", 20,20);
    // cv::moveWindow(
    //     "Display frame",
    //     1920 - window_width,
    //     1080 - window_height - 40);
    cv::imshow("Display frame", frame);
    //print height and width of iage
    cout<<"height: "<<frame.rows<<" width: "<<frame.cols<<endl;


    cv::waitKey(1);
}
    return 0;
}
   
        
    
    

void processFrame::sendExitMessage() {
    const char* exit_msg = "EXIT";
    if (clientSocket != -1 && clientSocket != -2) {
        cout<<"kapattı amk"<<endl;
        if (send(clientSocket, exit_msg, strlen(exit_msg), 0) == -1) {
                std::cerr << "Error sending message: " << strerror(errno) << std::endl;
        }
    }
    if (serverSocket != -1) {
        close(serverSocket);
        serverSocket = -1;
    }
    
}
void processFrame::sendFrame( const cv::Mat& frame) {
    std::vector<uchar> buffer;
    cv::imencode(".jpg", frame, buffer);
    int bufferSize = buffer.size();

    // Send the size of the buffer
    send(imageClientSocket, &bufferSize, sizeof(int), 0);

    // Send the buffer data
    send(imageClientSocket, buffer.data(), bufferSize, 0);
}
    
    
void processFrame::connect_image_server(){

  imageServerSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (imageServerSocket == -1) {
        cerr<<"socket creation error "<<strerror(errno) <<endl;
    }
    int enable=1;
    if (setsockopt(imageServerSocket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
        cerr<<"stsockopt(SO_REUSEADDR) failed"<<endl;


  // Set up server address
    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr =  htonl(INADDR_ANY); // Replace with desired port
    serverAddress.sin_port = htons(8080);

  // Bind the socket
    if (bind(imageServerSocket, (sockaddr*)&serverAddress, sizeof(serverAddress)) == -1) {
    // Handle error

        cout<<"binding error:"<< strerror(errno) <<endl;
        
   
    }
  

  // Listen for connections
    if (listen(imageServerSocket, 1) == -1) {
            cout<<"listen error "<<strerror(errno) <<endl;

        // Handle error
    }

    std::cout << "Server listening on port ..." << std::endl;

  // Accept a connection
    sockaddr_in clientAddress;
    socklen_t clientAddressSize = sizeof(clientAddress);
    while ((imageClientSocket = accept(imageServerSocket, (sockaddr*)&clientAddress, &clientAddressSize)) == -1) {
        std::cerr << "Client socket error: " << strerror(errno) << std::endl;
    }

    std::cout << "Client connected from " << inet_ntoa(clientAddress.sin_addr) << ":" << ntohs(clientAddress.sin_port) << std::endl;


}

void processFrame::connect_server(){
    

    
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        cerr<<"socket creation error "<<strerror(errno) <<endl;
    }
    int enable=1;
    if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
        cerr<<"stsockopt(SO_REUSEADDR) failed"<<endl;


  // Set up server address
    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr =  htonl(INADDR_ANY); // Replace with desired port
    serverAddress.sin_port = htons(1234);

  // Bind the socket
    if (bind(serverSocket, (sockaddr*)&serverAddress, sizeof(serverAddress)) == -1) {
    // Handle error

        cout<<"binding error:"<< strerror(errno) <<endl;
        
   
    }
  

  // Listen for connections
    if (listen(serverSocket, 1) == -1) {
            cout<<"listen error "<<strerror(errno) <<endl;

        // Handle error
    }

    std::cout << "Server listening on port ..." << std::endl;

  // Accept a connection
    sockaddr_in clientAddress;
    socklen_t clientAddressSize = sizeof(clientAddress);
    while ((clientSocket = accept(serverSocket, (sockaddr*)&clientAddress, &clientAddressSize)) == -1) {
        std::cerr << "Client socket error: " << strerror(errno) << std::endl;
    }

    std::cout << "Client connected from " << inet_ntoa(clientAddress.sin_addr) << ":" << ntohs(clientAddress.sin_port) << std::endl;

  

    sleep(2);
    while (true){
        char buffer[1024];
        memset(buffer, 0, 1024);

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(clientSocket, &readfds);

        struct timeval timeout;
        timeout.tv_sec = 0.5;
        timeout.tv_usec = 0;

        int activity = select(clientSocket + 1, &readfds, NULL, NULL, &timeout);

        if (activity < 0) {
            std::cerr << "Select error: " << strerror(errno) << std::endl;
        } else if (activity == 0) {
            std::cout << "Timeout: connect No data received for " << 0.5 << " seconds." << std::endl;
        }
        else{
            ssize_t bytesReceived = recv(clientSocket, buffer, 1024 - 1, 0);
            if (bytesReceived < 0) {
                std::cerr << "Error receiving message: " << strerror(errno) << std::endl;
            } else if (bytesReceived == 0) {
                std::cout << "Client disconnected." << std::endl;
            }
            std::cout << "Received echo: " << buffer << std::endl;

            if(strncmp(buffer,"Connection established from client.",1024)==0){
                break;
            }
   
        }
    }


}
