#ifndef MPPPERSONDET_H_
#define MPPPERSONDET_H_


#include <vector>
#include <string>
#include <utility>
#include <cmath>

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace dnn;

Mat getMediapipeAnchor();

class MPPersonDet {
public:

    Net net;
    string modelPath;
    Size inputSize;
    float scoreThreshold;
    float nmsThreshold;
    dnn::Backend backendId;
    dnn::Target targetId;
    int topK;
    Mat anchors;
public:
    MPPersonDet(string modPath, float nmsThresh = 0.5, float scoreThresh = 0.3, int tok=1 , dnn::Backend bId = DNN_BACKEND_DEFAULT, dnn::Target tId = DNN_TARGET_CPU) :
        modelPath(modPath), nmsThreshold(nmsThresh),
        scoreThreshold(scoreThresh), topK(tok),
        backendId(bId), targetId(tId)
    {
        this->inputSize = Size(224, 224);
        this->net = readNet(this->modelPath);
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
        this->anchors = getMediapipeAnchor();
    }

    pair<Mat, Size> preprocess(Mat img);
    

    Mat infer(Mat srcimg);
    

    Mat postprocess(vector<Mat> outputs, Size orgSize, Size padBias);


};


#endif