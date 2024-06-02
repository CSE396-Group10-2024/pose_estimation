#ifndef PROCESS_FRAME_H_
#define PROCESS_FRAME_H_



#include "MPPersonDet.h" // For MPPersonDet
#include "MPPose.h" // For MPPose
/* for general */
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <filesystem> // For std::filesystem
namespace cv 
{
    class Mat;
};
using namespace std;
/**
 * @brief This file is about analyzing the images from the specific source and camera.
 * This file includes all the steps of analyzing the image.
 * Before analyzing the image, initialize the image.
 * Then check if the image is initialized successfully by using the function 'Finalize'.
 * Next begin to analyze the images.
 * 
*/

class processFrame {
public:
string model = "../pose_estimation_mediapipe_2023mar.onnx";
    float confThreshold = 0.2f;
    float scoreThreshold = 0.5f;
    float nmsThreshold = 0.3f;
    int topK = 5000;
 
    int backendTargetid = 0;
    double angle_camera[8]; // angle check for users' pose from camera
    int Learner[12][2];
    std::filesystem::path txt_file_name_for_data;
    std::filesystem::path csv_file_name_for_data;
         MPPersonDet personDetector;
    MPPose poseDetector;

    void sendExitMessage();
   processFrame(string person_detection_path,string pose_detection_path);
    void set_engine(const std::string& txt_file_name_for_data, const std::string& csv_file_name_for_data, bool connect_to_server, const std::string &mode);
    int serverSocket = -1;
    int clientSocket = -2;
    int imageServerSocket = -2;
    int imageClientSocket = -2;
    int imageServerSocketPort=8080;
    int write_file_descriptor = -2;
    std::string mode;
    std::string label;
    std::ofstream outFile_;
    void connect_server();
    void connect_image_server();
    void sendFrame( const cv::Mat& frame);
    void change_mode(std::string mode);
    static int angle_check[8]; // comparing the angles of image pose and user's pose
    char directory_name[1024];
    
    int index;

    int32_t Process(cv::Mat& mat);
    void set_index(int index);
    int get_index();
};




#endif