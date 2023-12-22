#pragma once

#include "ofMain.h"
#include "ofxCvHaarFinder.h"
#include "ofxOpenCv.h"
#include "Model.h"
const std::string TENSORFLOW_MODEL_PATH = "C:/Users/plesk/Documents/coding/of_v0.11.2_vs2017_release/apps/myApps/cvEmotion/bin/data/model/tensorflow_model.pb";

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		ofxCvHaarFinder finder;
		Model model{ TENSORFLOW_MODEL_PATH };
		ofVideoGrabber grabber;
		ofxCvColorImage	colorImg;
		ofxCvGrayscaleImage bwImg;
		vector<string>emotion_prediction_name;
		vector<float>emotion_prediction_val;
};
