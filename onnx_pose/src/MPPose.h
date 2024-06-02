#ifndef MPPPOSE_H_
#define MPPPOSE_H_
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <opencv2/opencv.hpp>

#include <iostream>
using namespace std;
using namespace cv;
using namespace dnn;

Mat getMediapipeAnchor();

class MPPose {
public:
    Net net;
    string modelPath;
    Size inputSize;
    float confThreshold;
    dnn::Backend backendId;
    dnn::Target targetId;
    float personBoxPreEnlargeFactor;
    float personBoxEnlargeFactor;
    Mat anchors;


    MPPose(string modPath, float confThresh = 0.5, dnn::Backend bId = DNN_BACKEND_DEFAULT, dnn::Target tId = DNN_TARGET_CPU) :
        modelPath(modPath), confThreshold(confThresh),
        backendId(bId), targetId(tId)
    {
        this->inputSize = Size(256, 256);
        this->net = readNet(this->modelPath);
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
        this->anchors = getMediapipeAnchor();
        // RoI will be larger so the performance will be better, but preprocess will be slower.Default to 1.
        this->personBoxPreEnlargeFactor = 1;
        this->personBoxEnlargeFactor = 1.25;
    }

    tuple<Mat, Mat, float, Mat, Size> preprocess(Mat image, Mat person);
    

    tuple<Mat, Mat, Mat, Mat, Mat, float> infer(Mat image, Mat person)
   ;

    tuple<Mat, Mat, Mat, Mat, Mat, float> postprocess(vector<Mat> blob, Mat rotatedPersonBox, float angle, Mat rotationMatrix, Size padBias, Size imgSize)
   ;
   
};


#endif