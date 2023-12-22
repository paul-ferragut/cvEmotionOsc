#ifndef MODEL_H
#define MODEL_H

#include <opencv2/opencv.hpp>
#include <iostream>

//#include "Image.h"
#include "ofMain.h"
#include "ofxOpenCv.h"

/**
 * Model class that contains code to read the pretrained tensorflow model and allows
 * us to make predictions on new images
 */
class Model {
public:
    // Constructor reads in a pretrained (in python) tensorflow graph and weights (.pb file contains everything we need about the model)
    // Also initialises the mapping from class id to the string label (ie. happy, angry, sad etc)  
    Model(const string& model_filename);
    void modelSetup(bool useCuda);
    //Destructor
    ~Model() {};

    // Model inference function takes image input and outputs the prediction label and the probability
    vector<float> predict(vector<cv::Mat>image);
    vector<string>emotionList();


    vector<float>emotionValues;
    vector<string>emotionsNames;
private:
    // Neural network model
    cv::dnn::Net network;

    // Mapping of the class id to the string label
    map<int, string> classid_to_string;

};


#endif