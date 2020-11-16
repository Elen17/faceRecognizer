#include<iostream>
#include<opencv2/core/core.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include"opencv2/opencv.hpp"

using namespace std;
using namespace cv;



class WorkingWithFaces {
private:
    size_t imageCount = 0;

public:
    std::string path;

    WorkingWithFaces(std::string path) {
        this->path = path;

    }
   
    void getFacesFromVideo(Mat captureFrame, CascadeClassifier face_cascade); 
 
};

void WorkingWithFaces::getFacesFromVideo(Mat captureFrame, CascadeClassifier face_cascade) {
    
        Mat grayscaleFrame;
        //convert captured image to gray scale and equalize
        cvtColor(captureFrame, grayscaleFrame, COLOR_BGR2GRAY);
        equalizeHist(grayscaleFrame, grayscaleFrame);

        //create a vector array to store the face found
        std::vector<cv::Rect> faces;

        //find faces and store them in the vector array
        face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_SCALE_IMAGE, Size(30, 30));

        //draw a rectangle for all found faces in the vector array on the original image
        for (int i = 0; i < faces.size(); i++)
        {
            imwrite(this->path + "pic" + to_string(imageCount++) + ".jpg", Mat(captureFrame, faces[i]));
            Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
            Point pt2(faces[i].x, faces[i].y);
            rectangle(captureFrame, pt1, pt2, Scalar(0, 255, 0, 0), 1, 8, 0);
        }
        cv::imshow("", captureFrame);
        waitKey(10);
    

}