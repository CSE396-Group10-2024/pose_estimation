#include "MPPose.h"

#include <iostream>

#include <vector>
#include <string>
#include <utility>
#include <cmath>

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace dnn;
const long double _M_PI = 3.141592653589793238L;

tuple<Mat, Mat, float, Mat, Size> MPPose::preprocess(Mat image, Mat person)
    {
        /***
                Rotate input for inference.
                Parameters:
                  image - input image of BGR channel order
                  face_bbox - human face bounding box found in image of format [[x1, y1], [x2, y2]] (top-left and bottom-right points)
                  person_landmarks - 4 landmarks (2 full body points, 2 upper body points) of shape [4, 2]
                Returns:
                  rotated_person - rotated person image for inference
                  rotate_person_bbox - person box of interest range
                  angle - rotate angle for person
                  rotation_matrix - matrix for rotation and de-rotation
                  pad_bias - pad pixels of interest range
        */
        //  crop and pad image to interest range
        Size padBias(0, 0); // left, top
        Mat personKeypoints = person.colRange(4, 12).reshape(0, 4);
        Point2f midHipPoint = Point2f(personKeypoints.row(0));
        Point2f fullBodyPoint = Point2f(personKeypoints.row(1));
        // # get RoI
        double fullDist = norm(midHipPoint - fullBodyPoint);
        Mat fullBoxf,fullBox;
        Mat v1 = Mat(midHipPoint) - fullDist, v2 = Mat(midHipPoint);
        vector<Mat> vmat = { Mat(midHipPoint) - fullDist, Mat(midHipPoint) + fullDist };
        hconcat(vmat, fullBoxf);
        // enlarge to make sure full body can be cover
        Mat cBox, centerBox, whBox;
        reduce(fullBoxf, centerBox, 1, REDUCE_AVG, CV_32F);
        whBox = fullBoxf.col(1) - fullBoxf.col(0);
        Mat newHalfSize = whBox * this->personBoxPreEnlargeFactor / 2;
        vmat[0] = centerBox - newHalfSize;
        vmat[1] = centerBox + newHalfSize;
        hconcat(vmat, fullBox);
        Mat personBox;
        fullBox.convertTo(personBox, CV_32S);
        // refine person bbox
        Mat idx = personBox.row(0) < 0;
        personBox.row(0).setTo(0, idx);
        idx = personBox.row(0) >= image.cols;
        personBox.row(0).setTo(image.cols , idx);
        idx = personBox.row(1) < 0;
        personBox.row(1).setTo(0, idx);
        idx = personBox.row(1) >= image.rows;
        personBox.row(1).setTo(image.rows, idx);        // crop to the size of interest

        image = image(Rect(personBox.at<int>(0, 0), personBox.at<int>(1, 0), personBox.at<int>(0, 1) - personBox.at<int>(0, 0), personBox.at<int>(1, 1) - personBox.at<int>(1, 0)));
        // pad to square
        int top = int(personBox.at<int>(1, 0) - fullBox.at<float>(1, 0));
        int left = int(personBox.at<int>(0, 0) - fullBox.at<float>(0, 0));
        int bottom = int(fullBox.at<float>(1, 1) - personBox.at<int>(1, 1));
        int right = int(fullBox.at<float>(0, 1) - personBox.at<int>(0, 1));
        copyMakeBorder(image, image, top, bottom, left, right, BORDER_CONSTANT, Scalar(0, 0, 0));
        padBias = Point(padBias) + Point(personBox.col(0)) - Point(left, top);
        // compute rotation
        midHipPoint -= Point2f(padBias);
        fullBodyPoint -= Point2f(padBias);
        float radians = float(_M_PI / 2 - atan2(-(fullBodyPoint.y - midHipPoint.y), fullBodyPoint.x - midHipPoint.x));
        radians = radians - 2 * float(_M_PI) * int((radians + _M_PI) / (2 * _M_PI));
        float angle = (radians * 180 / float(_M_PI));
        //  get rotation matrix*
        Mat rotationMatrix = getRotationMatrix2D(midHipPoint, angle, 1.0);
        //  get rotated image
        Mat rotatedImage;
        warpAffine(image, rotatedImage, rotationMatrix, Size(image.cols, image.rows));
        //  get landmark bounding box
        Mat blob;
        Image2BlobParams paramPoseMediapipe;
        paramPoseMediapipe.datalayout = DNN_LAYOUT_NHWC;
        paramPoseMediapipe.ddepth = CV_32F;
        paramPoseMediapipe.mean = Scalar::all(0);
        paramPoseMediapipe.scalefactor = Scalar::all(1 / 255.);
        paramPoseMediapipe.size = this->inputSize;
        paramPoseMediapipe.swapRB = true;
        paramPoseMediapipe.paddingmode = DNN_PMODE_NULL;
        blob = blobFromImageWithParams(rotatedImage, paramPoseMediapipe); // resize INTER_AREA becomes INTER_LINEAR in blobFromImage
        Mat rotatedPersonBox = (Mat_<float>(2, 2) << 0, 0, image.cols, image.rows);

        return tuple<Mat, Mat, float, Mat, Size>(blob, rotatedPersonBox, angle, rotationMatrix, padBias);
    }


 tuple<Mat, Mat, Mat, Mat, Mat, float> MPPose::infer(Mat image, Mat person)
    {
        int h = image.rows;
        int w = image.cols;
        // Preprocess
        tuple<Mat, Mat, float, Mat, Size> tw;
        tw = this->preprocess(image, person);
        Mat inputBlob = get<0>(tw);
        Mat rotatedPersonBbox = get<1>(tw);
        float  angle = get<2>(tw);
        Mat rotationMatrix = get<3>(tw);
        Size padBias = get<4>(tw);

        // Forward
        this->net.setInput(inputBlob);
        vector<Mat> outputBlob;
        this->net.forward(outputBlob, this->net.getUnconnectedOutLayersNames());

        // Postprocess
        tuple<Mat, Mat, Mat, Mat, Mat, float> results;
        results = this->postprocess(outputBlob, rotatedPersonBbox, angle, rotationMatrix, padBias, Size(w, h));
        return results;// # [bbox_coords, landmarks_coords, conf]
    }

    tuple<Mat, Mat, Mat, Mat, Mat, float> MPPose::postprocess(vector<Mat> blob, Mat rotatedPersonBox, float angle, Mat rotationMatrix, Size padBias, Size imgSize)
    {
        float valConf = blob[1].at<float>(0);
        if (valConf < this->confThreshold)
            return tuple<Mat, Mat, Mat, Mat, Mat, float>(Mat(), Mat(), Mat(), Mat(), Mat(), valConf);
        Mat landmarks = blob[0].reshape(0, 39);
        Mat mask = blob[2];
        Mat heatmap = blob[3];
        Mat landmarksWorld = blob[4].reshape(0, 39);

        Mat deno;
        // recover sigmoid score
        exp(-landmarks.colRange(3, landmarks.cols), deno);
        divide(1.0, 1 + deno, landmarks.colRange(3, landmarks.cols));
        // TODO: refine landmarks with heatmap. reference: https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/blazepose_tfjs/detector.ts#L577-L582
        heatmap = heatmap.reshape(0, heatmap.size[0]);
        // transform coords back to the input coords
        Mat whRotatedPersonPbox = rotatedPersonBox.row(1) - rotatedPersonBox.row(0);
        Mat scaleFactor = whRotatedPersonPbox.clone();
        scaleFactor.col(0) /= this->inputSize.width;
        scaleFactor.col(1) /= this->inputSize.height;
        landmarks.col(0) = (landmarks.col(0) - this->inputSize.width / 2) * scaleFactor.at<float>(0);
        landmarks.col(1) = (landmarks.col(1) - this->inputSize.height / 2) * scaleFactor.at<float>(1);
        landmarks.col(2) = landmarks.col(2) * max(scaleFactor.at<float>(1), scaleFactor.at<float>(0));
        Mat coordsRotationMatrix;
        getRotationMatrix2D(Point(0, 0), angle, 1.0).convertTo(coordsRotationMatrix, CV_32F);
        Mat rotatedLandmarks = landmarks.colRange(0, 2) * coordsRotationMatrix.colRange(0, 2);
        hconcat(rotatedLandmarks, landmarks.colRange(2, landmarks.cols), rotatedLandmarks);
        Mat rotatedLandmarksWorld = landmarksWorld.colRange(0, 2) * coordsRotationMatrix.colRange(0, 2);
        hconcat(rotatedLandmarksWorld, landmarksWorld.col(2), rotatedLandmarksWorld);
        // invert rotation
        Mat rotationComponent  = (Mat_<double>(2, 2) <<rotationMatrix.at<double>(0,0), rotationMatrix.at<double>(1, 0), rotationMatrix.at<double>(0, 1), rotationMatrix.at<double>(1, 1));
        Mat translationComponent = rotationMatrix(Rect(2, 0, 1, 2)).clone();
        Mat invertedTranslation = -rotationComponent * translationComponent;
        Mat inverseRotationMatrix;
        hconcat(rotationComponent, invertedTranslation, inverseRotationMatrix);
        Mat center, rc;
        reduce(rotatedPersonBox, rc, 0, REDUCE_AVG, CV_64F);
        hconcat(rc, Mat(1, 1, CV_64FC1, 1) , center);
        //  get box center
        Mat originalCenter(2, 1, CV_64FC1);
        originalCenter.at<double>(0) = center.dot(inverseRotationMatrix.row(0));
        originalCenter.at<double>(1) = center.dot(inverseRotationMatrix.row(1));
        for (int idxRow = 0; idxRow < rotatedLandmarks.rows; idxRow++)
        {
            landmarks.at<float>(idxRow, 0) = float(rotatedLandmarks.at<float>(idxRow, 0) + originalCenter.at<double>(0) + padBias.width); // 
            landmarks.at<float>(idxRow, 1) = float(rotatedLandmarks.at<float>(idxRow, 1) + originalCenter.at<double>(1) + padBias.height); // 
        }
        // get bounding box from rotated_landmarks
        double vmin0, vmin1, vmax0, vmax1;
        minMaxLoc(landmarks.col(0), &vmin0, &vmax0);
        minMaxLoc(landmarks.col(1), &vmin1, &vmax1);
        Mat bbox = (Mat_<float>(2, 2) << vmin0, vmin1, vmax0, vmax1);
        Mat centerBox;
        reduce(bbox, centerBox, 0, REDUCE_AVG, CV_32F);
        Mat whBox = bbox.row(1) - bbox.row(0);
        Mat newHalfSize = whBox * this->personBoxEnlargeFactor / 2;
        vector<Mat> vmat(2);
        vmat[0] = centerBox - newHalfSize;
        vmat[1] = centerBox + newHalfSize;
        vconcat(vmat, bbox);
        // invert rotation for mask
        mask = mask.reshape(1, 256);
        Mat invertRotationMatrix = getRotationMatrix2D(Point(mask.cols / 2, mask.rows / 2), -angle, 1.0);
        Mat invertRotationMask;
        warpAffine(mask, invertRotationMask, invertRotationMatrix, Size(mask.cols, mask.rows));
        // enlarge mask
        resize(invertRotationMask, invertRotationMask, Size(int(whRotatedPersonPbox.at<float>(0)), int(whRotatedPersonPbox.at<float>(1))));
        // crop and pad mask
        int minW = -min(padBias.width, 0);
        int minH= -min(padBias.height, 0);
        int left = max(padBias.width, 0);
        int top = max(padBias.height, 0);
        Size padOver = imgSize - Size(invertRotationMask.cols, invertRotationMask.rows) - padBias;
        int maxW = min(padOver.width, 0) + invertRotationMask.cols;
        int maxH = min(padOver.height, 0) + invertRotationMask.rows;
        int right = max(padOver.width, 0);
        int bottom = max(padOver.height, 0);
        invertRotationMask = invertRotationMask(Rect(minW, minH, maxW - minW, maxH - minH)).clone();
        copyMakeBorder(invertRotationMask, invertRotationMask, top, bottom, left, right, BORDER_CONSTANT, Scalar::all(0));
        // binarize mask
        threshold(invertRotationMask, invertRotationMask, 1, 255, THRESH_BINARY);

        /* 2*2 person bbox: [[x1, y1], [x2, y2]]
        # 39*5 screen landmarks: 33 keypoints and 6 auxiliary points with [x, y, z, visibility, presence], z value is relative to HIP
        # Visibility is probability that a keypoint is located within the frame and not occluded by another bigger body part or another object
        # Presence is probability that a keypoint is located within the frame
        # 39*3 world landmarks: 33 keypoints and 6 auxiliary points with [x, y, z] 3D metric x, y, z coordinate
        # img_height*img_width mask: gray mask, where 255 indicates the full body of a person and 0 means background
        # 64*64*39 heatmap: currently only used for refining landmarks, requires sigmod processing before use
        # conf: confidence of prediction*/
        return tuple<Mat , Mat, Mat, Mat, Mat, float>(bbox, landmarks, rotatedLandmarksWorld, invertRotationMask, heatmap, valConf);
    }

