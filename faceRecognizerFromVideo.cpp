//#include<iostream>
//#include<stdio.h>
//#include<opencv2/core/core.hpp>
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include"opencv2/opencv.hpp"
//#include<opencv2/imgcodecs/imgcodecs.hpp>
//#include "opencv2/include_seeta/face_detection.h"
//#include<opencv2/include_seeta/face_alignment.h>
//#include"opencv2/include_seeta/recognizer.h"
//#include"opencv2/include_seeta/face_identification.h"
//#include <opencv2/videoio.hpp>
//
//using namespace std;
//using namespace cv;
//
//int fmain(int argc, const char** argv) {
//	CascadeClassifier face_cascade;
//	//use the haarcascade_frontalface_alt.xml library
//	face_cascade.load("C:\\Users\\Elen\\Desktop\\Project2\\Project2\\haarcascade_frontalface_alt.xml");
//
//	//setup video capture device and link it to the first capture device
//	VideoCapture captureDevice;
//	captureDevice.open(0);
//
//	//setup image files used in the capture process
//	Mat captureFrame;
//	Mat grayscaleFrame;
//
//	//create a window to present the results
//	namedWindow("Output", 1);
//	int imageCount = 1;
//	//create a loop to capture and find faces
//	while (true)
//	{
//		//capture a new image frame
//		captureDevice >> captureFrame;
//
//		//convert captured image to gray scale and equalize
//		cvtColor(captureFrame, grayscaleFrame, COLOR_BGR2GRAY);
//		equalizeHist(grayscaleFrame, grayscaleFrame);
//
//		//create a vector array to store the face found
//		std::vector<Rect> faces;
//
//		//find faces and store them in the vector array
//		face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_SCALE_IMAGE, Size(30, 30));
//
//		//draw a rectangle for all found faces in the vector array on the original image
//		for (int i = 0; i < faces.size(); i++)
//		{
//			imwrite("C:\\Users\\Elen\\Desktop\\Project2\\Project2\\newFaces\\pic" + to_string(imageCount++) + ".jpg", Mat(captureFrame, faces[i]));
//			Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
//			Point pt2(faces[i].x, faces[i].y);
//			rectangle(captureFrame, pt1, pt2, Scalar(0, 255, 0, 0), 1, 8, 0);
//		}
//		//
//		////print the output
//		//imshow("Output", captureFrame);
//		//pause for 400ms//it pics every 40ms
//		waitKey(400);
//	}
//	return 0;
//
//}