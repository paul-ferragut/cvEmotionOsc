#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	finder.setup("model/haarcascade_frontalface_alt2.xml");
	//finder.findHaarObjects(img);
	if (!settings.load("settings.xml")) {
		ofLogError() << "Couldn't load file";
	}
	model.modelSetup(true);
	grabber.setDeviceID(settings.getValue("deviceID",  0,0));
	grabber.initGrabber(640, 480);
	colorImg.allocate(640, 480);	
	bwImg.allocate(640, 480);

	ofSetFrameRate(60); // run at 60 fps
	ofSetVerticalSync(true);



	// open an outgoing connection to HOST:PORT
	string host = settings.getValue("host", "localhost", 0);
	//host = "\"" + host + "\"";
	int port = settings.getValue("port", 12345,0);
	cout << "host"<< host << endl;
	cout << "port" << port << endl;
	sender.setup(host, port);

	senderSpout.init("CamEmotions");

}

//--------------------------------------------------------------
void ofApp::update(){


	grabber.update();
	if (grabber.isFrameNew()) {

		senderSpout.send(grabber.getTexture());

		//get the ofPixels and convert to an ofxCvColorImage
		auto pixels = grabber.getPixels();
		colorImg.setFromPixels(pixels);
	
		bwImg = colorImg;
		finder.findHaarObjects(bwImg);

		

		if (finder.blobs.size() > 0) {

			//Find larger blob
			int largerRect = 0;
			int prevSize = 0;
			for (unsigned int i = 0; i < finder.blobs.size(); i++) {

				CvRect rectOF;
				rectOF.x = finder.blobs[i].boundingRect.x;
				rectOF.y = finder.blobs[i].boundingRect.y;
				rectOF.width = finder.blobs[i].boundingRect.width;
				rectOF.height = finder.blobs[i].boundingRect.height;
				int sizeR = rectOF.width * rectOF.height;
				if (prevSize < sizeR) {
					largerRect = i;
				}
				prevSize = sizeR;
			}

			// Make Prediction
			vector< cv::Mat> matCv;
			for (unsigned int i = 0; i < finder.blobs.size(); i++) {

				if (i == largerRect) {
					CvRect rectOF;
					rectOF.x = finder.blobs[i].boundingRect.x;
					rectOF.y = finder.blobs[i].boundingRect.y;
					rectOF.width = finder.blobs[i].boundingRect.width;
					rectOF.height = finder.blobs[i].boundingRect.height;

					cv::Mat t = colorImg.getCvMat();
					cv::Mat c = t(rectOF);
					cv::Mat processed_image;
					cv::Mat gray_image;
					cv::cvtColor(c, gray_image, cv::COLOR_BGR2GRAY);
					cv::resize(gray_image, processed_image, cv::Size(48, 48));
					// Convert image pixels from between 0-255 to 0-1
					processed_image.convertTo(processed_image, CV_32FC3, 1.f / 255);
					matCv.push_back(processed_image);
				}
			}
			emotion_prediction_val= model.predict(matCv);
			emotion_prediction_name = model.emotionList();



			for (int i = 0; i < emotion_prediction_val.size(); i++) {
				//ofDrawBitmapString(model.probS[i], 20, 480 + i * 40);
			ofxOscMessage m;
			m.setAddress("/emotion/"+ emotion_prediction_name[i]);
			m.addFloatArg(emotion_prediction_val[i]);
			sender.sendMessage(m, false);
			}

			//vector<string>emotion_prediction = model.predict(matCv);
			// Add prediction text to the output video frame
			//for (int i = 0; i < emotion_prediction.size(); i++) {
			//	cout << emotion_prediction[i] << endl;
			//}
		}
	}
	
	
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofFill();
	grabber.draw(0,0);
	ofNoFill();
	for (unsigned int i = 0; i < finder.blobs.size(); i++) {
		ofRectangle cur = finder.blobs[i].boundingRect;
		ofDrawRectangle(cur.x, cur.y, cur.width, cur.height);
		
	}

	//for (int i = 0; i < emotion_prediction.size(); i++) {
	//	ofDrawBitmapString(emotion_prediction[i], finder.blobs[i].boundingRect.x, finder.blobs[i].boundingRect.y);
	//}

	for (int i = 0; i < emotion_prediction_val.size(); i++) {
		//ofDrawBitmapString(model.probS[i], 20, 480 + i * 40);
		ofFill();
		ofSetColor(255, 0, 0, 255);
		ofDrawRectangle(10, 490 + i * 40, emotion_prediction_val[i]*200, 20);
		ofNoFill();
		ofSetColor(255,255, 255, 255);
		ofDrawRectangle(10, 490 + i * 40,200, 20);
	}

	for (int i = 0; i < emotion_prediction_name.size(); i++) {
		ofDrawBitmapString(emotion_prediction_name[i], 20, 480 + i * 40);
		//ofDrawRectangle(10, 490 + i * 40, model.probF[i], 20);
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
