#pragma once

#include "ofxSpout.h"
#include "ofMain.h"

#include "ofxCvHaarFinder.h"
#include "ofxOpenCv.h"
#include "Model.h"
#include "ofxGui.h"
#include "ofxXmlSettings.h"
#include "ofxOsc.h"


// send host (aka ip address)
#define HOST "localhost"

/// send port
#define PORT 12345

//const std::string TENSORFLOW_MODEL_PATH = "C:/Users/Paul/Documents/coding/of_v0.12.0_vs_release/of_v0.12.0_vs_release/apps/myApps/cvEmotionOsc/bin/data/model/tensorflow_model.pb";
const std::string TENSORFLOW_MODEL_PATH = ".//data//model//tensorflow_model.pb";


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

		ofxPanel gui;
		ofxXmlSettings settings;

		ofxOscSender sender;

		ofxSpout::Sender senderSpout;

};
