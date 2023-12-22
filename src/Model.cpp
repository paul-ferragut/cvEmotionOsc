#include <opencv2/opencv.hpp>

#include "Model.h"

Model::Model(const string& model_filename) 
    : network(cv::dnn::readNet(model_filename)), // Load the tensorflow model 
      classid_to_string({{0, "Angry"}, 
                         {1, "Disgust"}, 
                         {2, "Fear"}, 
                         {3, "Happy"}, 
                         {4, "Sad"}, 
                         {5, "Surprise"}, 
                         {6, "Neutral"}}) // Create a map from class id to the class labels
{}



void Model::modelSetup(bool useCuda)
{

    emotionValues.clear();
    emotionsNames.clear();
    if (useCuda) {
        cout << "Attempt Yolo with CUDA\n";
        network.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        network.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        cout << "Running Yolo on CPU with OpenCL if available\n";
        network.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        network.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    }
}

vector<float> Model::predict(vector<cv::Mat>image) {
    // this takes the region of interest image and then runs model inference
   vector<cv::Mat> roi_image = image;// image.getCvMat();

  //  vector<float> emotion_prediction;

    if (roi_image.size() > 0) { 
        for (int i=0; i < roi_image.size(); i++) {
            // Convert to blob
            cv::Mat blob = cv::dnn::blobFromImage(roi_image[i]);

            // Pass blob to network
            this->network.setInput(blob);

            // Forward pass on network    
            cv::Mat prob = this->network.forward();

           
            // Sort the probabilities and rank the indicies
           // cv::Mat sorted_probabilities;
           // cv::Mat sorted_ids;
            //cv::sort(prob.reshape(1, 1), sorted_probabilities, cv::SORT_DESCENDING);
           // cv::sortIdx(prob.reshape(1, 1), sorted_ids, cv::SORT_DESCENDING);

            // Sort the probabilities and rank the indicies
            cv::Mat unsorted_probabilities;
            cv::Mat unsorted_ids;
            unsorted_probabilities = prob.reshape(1, 1);

            //cout<<unsorted_probabilities.size().width<<endl;
            //unsorted_ids = prob.reshape(1, 1);
            //prob.reshape(1, 1).copyTo(unsorted_probabilities);
           // prob.reshape(1, 1).clone()
            //cv::write(prob.reshape(1, 1), unsorted_probabilities);
            //cv::copyTo(prob.reshape(1, 1), unsorted_probabilities);
           // cv::sort(prob.reshape(1, 1), unsorted_probabilities, cv::SORT_EVERY_ROW);
           // cv::sortIdx(prob.reshape(1, 1), unsorted_ids, cv::SORT_EVERY_ROW);
           // unsorted_probabilities.total()
            emotionsNames.clear();
            emotionValues.clear();
            for (int i = 0; i< unsorted_probabilities.size().width; i++) {
                //cout << i<< " unsorted prob" << unsorted_probabilities.at<float>(i) << endl;
                //cout << i << " emotion prob" << this->classid_to_string.at(unsorted_ids.at<int>(i)) << endl;
                emotionValues.push_back(unsorted_probabilities.at<float>(i));
                emotionsNames.push_back(this->classid_to_string.at(i));//unsorted_ids.at<int>(i)
                
            }

            // Get top probability and top class id
            //float top_probability = sorted_probabilities.at<float>(0);
           // int top_class_id = sorted_ids.at<int>(0);

            

            // Map classId to the class name string (ie. happy, sad, angry, disgust etc.)
            //std::string class_name = this->classid_to_string.at(top_class_id);

            // Prediction result string to print
           // std::string result_string = class_name + ": " + std::to_string(top_probability * 100) + "%";

            // Put on end of result vector
            //emotion_prediction.push_back(emotionValues)
           // emotion_prediction.push_back(emotionValues);

        }
    }

    return emotionValues;

}

vector<string> Model::emotionList()
{

    return emotionsNames;
}



