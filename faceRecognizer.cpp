//#include<iostream>
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
//using namespace std;
//std::string MODEL_DIR = "C:\\Users\\Elen\\Downloads\\SeetaFaceEngine-master\\FaceIdentification\\model\\";
//std::string DATA_DIR = "C:\\Users\\Elen\\source\\repos\\Project5\\Project5\\data\\images\\";
//std::string SRC_DIR = DATA_DIR + "videoInput\\";
//using namespace seeta;
//using namespace cv;
//// Initialize face detection model 
// seeta::FaceDetection detector((MODEL_DIR + "seeta_fd_frontal_v1.0.bin").c_str());
//
//    // Initialize face alignment model 
// seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());
//
//    // Initialize face identification model 
// seeta::FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
//
//void getImageFeatures(const string& imgDirectory, float* features, cv::Mat* images, int index = -1) {
//   
//    //std::cout << imgDirectory << endl;
//    cv::Mat img = cv::imread(imgDirectory, 1);
//    cv::Mat img_gray = cv::imread(imgDirectory, 0);
//    seeta::ImageData img_gray_data(img_gray.cols, img_gray.rows, img_gray.channels());
//    img_gray_data.data = img_gray.data;
//    std::vector<seeta::FaceInfo> faces = detector.Detect(img_gray_data);
//    int32_t gallery_face_num = static_cast<int32_t>(faces.size());
//    seeta::FacialLandmark points[5];
//    point_detector.PointDetectLandmarks(img_gray_data, faces[0], points);
//    seeta::ImageData img_data(img.cols, img.rows, img.channels());
//    img_data.data = img.data;
//    face_recognizer.ExtractFeatureWithCrop(img_data, points, features);
//    if (images != nullptr && index != -1) {
//        int32_t num_face = static_cast<int32_t>(faces.size());
//        cv::Rect face_rect;
//        for (int32_t i = 0; i < num_face; i++) {
//            face_rect.x = faces[i].bbox.x;
//            face_rect.y = faces[i].bbox.y;
//            face_rect.width = faces[i].bbox.width;
//            face_rect.height = faces[i].bbox.height;
//            cv::rectangle(img, face_rect, CV_RGB(0, 20, 0), 2, 8, 0);
//        }
//        for (int i = 0; i < 5; i++)
//        {
//            cv::circle(img, cv::Point(points[i].x, points[i].y), 2, CV_RGB(255, 20, 0), cv::FILLED);
//        }
//        images[index] = img;
//    }
// 
//}
//
//void getInput() {
//
//}
//int fmain() {
//    detector.SetMinFaceSize(40);
//    detector.SetScoreThresh(2.f);
//    detector.SetImagePyramidScaleFactor(0.8f);
//    detector.SetWindowStep(4, 4); 
//    
//    //capture video and get photo
//    //setup video capture device and link it to the first capture device
//    VideoCapture captureDevice;
//    captureDevice.open(0);
//
//    //emmaWatson package
//    vector<cv::String> baseDir1;
//    cv::glob("C:\\Users\\Elen\\source\\repos\\Project5\\Project5\\data\\images\\emmaWatson\\*.jpg", baseDir1, false);
//    size_t count1 = baseDir1.size(); //number of png files in images folder
//
//    cv::Mat* emma = new cv::Mat[count1];
//    for (size_t i = 0; i < count1; i++)
//    {
//        *(emma + i) = cv::imread(baseDir1[i]);
//    }
//    //robertDeNiro package
//    vector<cv::String> baseDir2;
//    cv::glob("C:\\Users\\Elen\\source\\repos\\Project5\\Project5\\data\\images\\robertDeNiro\\*.jpg", baseDir2, false);
//    size_t count2 = baseDir2.size(); //number of png files in images folder
//
//    cv::Mat* robert = new cv::Mat[count2];
//    for (size_t i = 0; i < count2; i++)
//    {
//        *(robert + i) = cv::imread(baseDir2[i]);
//    }
//
//    //freeman package
//    vector<cv::String> baseDir3;
//    cv::glob("C:\\Users\\Elen\\source\\repos\\Project5\\Project5\\data\\images\\kingsley\\*.jpg", baseDir3, false);
//    size_t count3 = baseDir3.size(); //number of jpg files in images folder
//
//    cv::Mat* kingsley = new cv::Mat[count3];
//    for (size_t i = 0; i < count3; i++)
//    {
//        *(kingsley + i) = cv::imread(baseDir3[i]);
//        //cv::imshow("", *(kingsley + i));
//        //cv::waitKey(1);
//    }
//
//    vector<cv::String> freemanDir;
//    cv::glob("C:\\Users\\Elen\\source\\repos\\Project5\\Project5\\data\\images\\morgan\\*.jpg", freemanDir, false);
//    size_t count4 = freemanDir.size(); //number of jpg files in images folder
//
//    cv::Mat* freeman = new cv::Mat[count4];
//    for (size_t i = 0; i < count4; i++)
//    {
//        *(freeman + i) = cv::imread(freemanDir[i]);
//        //cv::imshow("", *(kingsley + i));
//        //cv::waitKey(1);
//    }
//
//    //load inputImages
//    vector<cv::String> inputImagesDir;
//    cv::glob(DATA_DIR + "input\\*.jpg", inputImagesDir, false);
//
//    size_t count_input = inputImagesDir.size(); //number of png files in images folder
//    cv::Mat* inputImages = new cv::Mat[count_input];
//    for (size_t j = 0; j < count_input; j++)
//    {
//        *(inputImages + j) = cv::imread(inputImagesDir[j]);
//
//    }
//
//
//    std::vector<float[2048]> gallery_features_first(count1);// = new std::vector<>(count1);
//    for (size_t i = 0; i < count1; i++)
//    {
//        getImageFeatures(baseDir1[i], gallery_features_first[i], emma ,i);
//        //cv::imshow("", *(emma + i)); 
//        //cv::waitKey();
//
//    }      
//
//    std::vector<float[2048]> gallery_features_sec(count2);// = new std::vector<>(count1);
//    for (size_t i = 0; i < count2; i++)
//    {
//        getImageFeatures(baseDir2[i], gallery_features_sec[i], robert, i);
//    }
//
//    std::vector<float[2048]> gallery_features_king(count3);// = new std::vector<>(count1);
//    for (size_t i = 0; i < count3; i++)
//    {
//        getImageFeatures(baseDir3[i], gallery_features_king[i], kingsley, i);
//        //cv::imshow("", *(emma + i)); 
//        //cv::waitKey();
//
//    }
//    std::vector<float[2048]> gallery_features_freeman(count4);// = new std::vector<>(count1);
//    for (size_t i = 0; i < count4; i++)
//    {
//        getImageFeatures(freemanDir[i], gallery_features_freeman[i], freeman, i);
//        
//    }
//
//    //std::cout <<"Emma Watson"<<"         "<<"Robert De Niro"<<"         "<< "Ben Kingsley"<<"         "<<"Morgan Freeman\n";
//
//    for (size_t i = 0; i < count_input; i++)
//    {
//        float  input_features[2048];
//        getImageFeatures(inputImagesDir[i], input_features, inputImages, i);
//        float sim1 = 0, sim2 = 0, sim3 = 0, sim4 = 0;
//        
//      
//
//        for (size_t j = 0; j < count1; j++)
//        {
//            sim1 += face_recognizer.CalcSimilarity(gallery_features_first[j], input_features);
//        }
//
//        for (size_t j = 0; j < count2; j++)
//        {
//            sim2 += face_recognizer.CalcSimilarity(gallery_features_sec[j], input_features);
//        }
//          for (size_t j = 0; j < count3; j++)
//        {
//            sim3 += face_recognizer.CalcSimilarity(gallery_features_king[j], input_features);
//        }
//          for (size_t j = 0; j < count4; j++)
//          {
//              sim4 += face_recognizer.CalcSimilarity(gallery_features_freeman[j], input_features);
//          }
//
//          cv::namedWindow("Input", cv::WINDOW_GUI_NORMAL);
//          cv::resizeWindow("Input", 150, 150);
//          cv::imshow("Input", inputImages[i]);
//          cv::waitKey(0);
//          cv::destroyWindow("Input");
//
//        std::cout << i <<":  "<< sim1/count1 << "         " << sim2/count2<< "         " << sim3/count3 <<"         "<< sim4/count4 << "\n";
//        
//    }
//    return 0;
//    
//   // system("pause");
//
//}
