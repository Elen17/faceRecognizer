#include<iostream>
#include<opencv2/core/core.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include"opencv2/opencv.hpp"
#include<opencv2/imgcodecs/imgcodecs.hpp>
#include "opencv2/include_seeta/face_detection.h"
#include<opencv2/include_seeta/face_alignment.h>
#include"opencv2/include_seeta/recognizer.h"
#include"opencv2/include_seeta/face_identification.h"
#include <opencv2/videoio.hpp>
#include<thread>
#include<cmath>
#include <utility>
#include <chrono>
#include<future>
#include <fstream>
#include "../../../../Downloads/SVMClassifier.h"

using namespace std;
std::string MODEL_DIR = "C:\\Users\\Elen\\Downloads\\SeetaFaceEngine-master\\FaceIdentification\\model\\";
std::string DATA_DIR = "C:\\Users\\Elen\\source\\repos\\Project5\\Project5\\data\\images\\";
std::string SRC_DIR = "C:\\Users\\Elen\\source\\repos\\Project5\\Project5\\data\\images\\videoInput\\";
std::string DB_FEATURE_FILE = "C:\\Users\\Elen\\Desktop\\features.txt";

using namespace seeta;
using namespace cv;
// Initialize face detection model 
seeta::FaceDetection detector((MODEL_DIR + "seeta_fd_frontal_v1.0.bin").c_str());

// Initialize face alignment model 
seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());

// Initialize face identification model 
seeta::FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
int imageCount = 1;
size_t current = 1;

bool getFacesFromVideo(Mat captureFrame, CascadeClassifier face_cascade) {

    bool result = false;
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

        cv::imwrite(SRC_DIR + "pic" + to_string(imageCount++) + ".jpg", Mat(captureFrame, faces[i]));
        Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        Point pt2(faces[i].x, faces[i].y);
        rectangle(captureFrame, pt1, pt2, Scalar(0, 255, 0, 0), 1, 8, 0);
        result = true;
    }
  /*  if (result) {
        cv::imshow("", captureFrame);
        waitKey(5);
    }*/

    return result;

}

const int FEATURE_DIMENSION = 2048;
const int PEOPLE_NUMBER = 10;
const int NUMBER_OF_PICTURES_FROM_EVERY_PERSON = 3;

void getBaseFeatures(vector<vector<float[FEATURE_DIMENSION]>> &dbFeatures,  String filePath) {
    fstream fs;
    fs.open(filePath, ios::in);
    //vector < vector<float[2048]>> floatVec = vector<vector<float[2048]>>();
    dbFeatures = vector<vector<float[FEATURE_DIMENSION]>>();
    string strFloat;
    float fNum;
    int counter = 0;
    int row = 0;
    int column = 0;
    while (counter != PEOPLE_NUMBER) {

        dbFeatures.push_back(std::vector<float[FEATURE_DIMENSION]>());
        dbFeatures.at(0) = vector<float[FEATURE_DIMENSION]>(NUMBER_OF_PICTURES_FROM_EVERY_PERSON);

        while (row != NUMBER_OF_PICTURES_FROM_EVERY_PERSON && getline(fs, strFloat)) {

            std::stringstream  linestream(strFloat);
            while (linestream >> fNum)
            {
                dbFeatures.at(counter).at(row)[column] = fNum;
                ++column;
                if (column == FEATURE_DIMENSION) {
                    column = 0;
                    ++row;
                    if (row == NUMBER_OF_PICTURES_FROM_EVERY_PERSON &&  ++counter != 10) {
                        row = 0;
                        //if counter  = 10 after incrementation, than break
                        dbFeatures.push_back(std::vector<float[FEATURE_DIMENSION]>());
                        dbFeatures.at(counter) = vector<float[FEATURE_DIMENSION]>(NUMBER_OF_PICTURES_FROM_EVERY_PERSON);
                    }
                    if (counter == PEOPLE_NUMBER) {
                        break;
                    }
                }

                if (counter == PEOPLE_NUMBER) {
                    break;
                }

            }

        }

    }
}

void getImageFeatures(const string& imgDirectory, float* features, cv::Mat* images, int index = -1) {

    cv::Mat img = cv::imread(imgDirectory, 1);
    cv::Mat img_gray = cv::imread(imgDirectory, 0);

    seeta::ImageData img_gray_data(img_gray.cols, img_gray.rows, img_gray.channels());
    img_gray_data.data = img_gray.data;
    std::vector<seeta::FaceInfo> faces = detector.Detect(img_gray_data);
    int32_t gallery_face_num = static_cast<int32_t>(faces.size());
    seeta::FacialLandmark points[5];
    point_detector.PointDetectLandmarks(img_gray_data, faces[0], points);
    seeta::ImageData img_data(img.cols, img.rows, img.channels());
    img_data.data = img.data;
    
    face_recognizer.ExtractFeatureWithCrop(img_data, points, features);
    if (images != nullptr && index != -1) {
        int32_t num_face = static_cast<int32_t>(faces.size());
        cv::Rect face_rect;
        for (int32_t i = 0; i < num_face; i++) {
            face_rect.x = faces[i].bbox.x;
            face_rect.y = faces[i].bbox.y;
            face_rect.width = faces[i].bbox.width;
            face_rect.height = faces[i].bbox.height;
            cv::rectangle(img, face_rect, CV_RGB(0, 20, 0), 2, 8, 0);
        }
        for (int i = 0; i < 5; i++)
        {
            cv::circle(img, cv::Point(points[i].x, points[i].y), 2, CV_RGB(255, 20, 0), cv::FILLED);
        }
        images[index] = img;
        
    }

}


void normalizeFeatures(float* features) {
    float* featuresCopy = features;
    int quadSum = 0;
    for (size_t i = 0; i < 2048; ++i) {
        quadSum += pow(*(featuresCopy + i), 2);
    }
    float sqrtQuadSum = sqrt(quadSum);
    for (size_t i = 0; i < 2048; i++) {
        *(features + i) /= sqrtQuadSum;
    }
}

//used while getting new faces features
std::vector<float[2048]> calcBaseFeauture(string dirPath, int numberOfFiles, String format) {
    vector<cv::String> baseDir;
    cv::glob(dirPath + "*." +  format, baseDir, false);
    size_t size = baseDir.size(); //number of pmg files in images folder


    cv::Mat* images = new cv::Mat[numberOfFiles];
    for (size_t i = 0; i < numberOfFiles; i++)
    {
        *(images + i) = cv::imread(baseDir[i]);
    }

    std::vector<float[2048]> gallery_features(numberOfFiles);
    for (size_t i = 0; i < numberOfFiles; i++)
    {
        getImageFeatures(baseDir[i], gallery_features[i], images, i);
        normalizeFeatures(&gallery_features[i][0]);

    }
    return gallery_features;

}

std::vector<float[2048]> getFeature(string dirPath) {
    vector<cv::String> baseDir;
    std::string picName = dirPath + "pic" + to_string(current) + ".jpg";
    cv::glob(picName, baseDir, false);
    //size_t size = baseDir.size(); //number of png files in images folder

    cv::Mat image = cv::imread(picName);

    std::vector<float[2048]> input(1);
    getImageFeatures(baseDir[0], input[0], &image, 0);
    ++current;
    normalizeFeatures(&(input[0][0]));
    return input;
}


std::string getName(int index) {
    switch (index) {
       
        case 1: {
            return "Emma";
        }
        case 2: {
            return "Lusine";
        }
        case 3: {
            return "Elen";
        }

        case 4: {
            return "Ani";
        }
        case 5: {
            return "Robert";
        }
        case 6: {
            return "Ben";
        }
        case 7: {
            return "Kate";
        }
        case 8: {
            return "Orlando";
        }
        case 9: {
            return "Johnny";
        }
        case 10: {
            return "Colin";
        }
        default: {
            return "";
        }

    }

}



Ptr<ml::SVM>/*void*/ getSVM(/*const*/ vector<vector<float[FEATURE_DIMENSION]/***/>>& peopleFeaturesArray) {
    Mat trainData;
    Mat labelsMat;
    for (int i = 0; i < PEOPLE_NUMBER; ++i) {
        for (int j = 0; j < NUMBER_OF_PICTURES_FROM_EVERY_PERSON; ++j) {
            labelsMat.push_back(i + 1);
            //normalizeFeature(peopleFeaturesArray[i][j]);
            Mat temp(1, FEATURE_DIMENSION/**/, CV_32F, peopleFeaturesArray[i][j]);

             trainData.push_back(temp);
        }
    }
    
    // Train the SVM
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setC(1);
    svm->setKernel(ml::SVM::LINEAR);
    //svm->setGamma(0.5);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS, 1000, 1e-6));

    Ptr<ml::TrainData> td = ml::TrainData::create(trainData, ml::ROW_SAMPLE, labelsMat);
    svm->train(td);
    bool trained = svm->train(ml::TrainData::create(trainData, ml::ROW_SAMPLE, labelsMat));

    cout << "\nSVM is " << (trained ? "" : "not ") << "trained " << endl;  
    svm->save("C:\\Users\\Elen\\Desktop\\trainFile.xml");

    while (!(svm->isTrained())) {}
    return svm;
}

int getLabel(float *testFeautures, Ptr<ml::SVM> svm) {
    
    Mat testData(1, 2048, CV_32F, testFeautures);/*1 -> testing one image, 2048 -> feature vectors length*/

    return svm->predict(testData);
}

vector<vector<float[2048]>> run() {
    vector<vector<float[2048]>> peopleFeatures;
    for (int i = 1; i <= PEOPLE_NUMBER; ++i) {
        std::cout << DATA_DIR + "s" + std::to_string(i) + "\\ \n";
        peopleFeatures.push_back(calcBaseFeauture(DATA_DIR + "s" + std::to_string(i) + "\\", NUMBER_OF_PICTURES_FROM_EVERY_PERSON,"pgm"));
    }
    SVMClassifier svmClassifier(PEOPLE_NUMBER, NUMBER_OF_PICTURES_FROM_EVERY_PERSON, FEATURE_DIMENSION);
    //	if (!svmClassifier.loadClassifier("svm1")) {
    cout << "Classifier not found.\nGenerating new classifier...\n";
    svmClassifier.setTraningData(peopleFeatures);
    svmClassifier.setTraningParameters();
    svmClassifier.train();
    svmClassifier.saveClassifier("svm1");
    svmClassifier.saveClassifier("C:\\Users\\Elen\\Desktop\\svm1.xml");
    //svmClassifier.loadClassifier("C:\\Users\\Elen\\Desktop\\svm1.xml");

    for (int j = 0; j < peopleFeatures.size(); ++j) {
        int count = 0;
        for (int i = 0; i < peopleFeatures[0].size(); ++i) {
            if (svmClassifier.predict(peopleFeatures[j][i]) == j + 1) {
                cout << "\nPredicted " << svmClassifier.predict(peopleFeatures[j][i]) << "\n";
                ++count;
            }
        }
        cout << "On Person" << j + 1 << "  data predicted : " << count << "/" << NUMBER_OF_PICTURES_FROM_EVERY_PERSON<< "\n================================================\n";
    }
    return peopleFeatures;
}

int main(int argc, char** argv) {
    CascadeClassifier face_cascade;
    //use the haarcascade_frontalface_alt.xml library
    face_cascade.load("C:\\Users\\Elen\\Desktop\\Project2\\Project2\\haarcascade_frontalface_alt.xml");

    //getting dataBase features
    size_t dirNumbers = PEOPLE_NUMBER;
    //contains paths of all persons in database
    std::vector<string> dirPaths;
    for (size_t i = 0; i < dirNumbers; i++)
    {
        dirPaths.push_back(DATA_DIR + "person" + std::to_string(i) + "\\");
       
    }


    // when new photos were added used to write their features in file

     /*fstream file;
    file.open(filePath);*/
    std::vector<vector<float[2048]>> dataBaseFeatures;

    for (size_t i = 0; i < dirNumbers; i++)
    {
        vector<float[2048]> feature = calcBaseFeauture(dirPaths.at(i), NUMBER_OF_PICTURES_FROM_EVERY_PERSON, "jpg");
        dataBaseFeatures.push_back(calcBaseFeauture(dirPaths.at(i), NUMBER_OF_PICTURES_FROM_EVERY_PERSON, "jpg"));

        //for (size_t k = 0; k < feature.size(); k++) {

        //    for (size_t j = 0; j < 2048; j++)
        //    {
        //        //file << deature.at(i).at(k)[j] << " ";
        //        std::cout << feature.at(k)[j] << " ";
        //    }
        //    std::cout << "";
        //}
    

    }

    //getBaseFeatures(dataBaseFeatures, DB_FEATURE_FILE);
    //dataBaseFeatures = run();

    auto svm = getSVM((dataBaseFeatures));

    //setup video capture device and link it to the first capture device
    VideoCapture captureDevice;
    captureDevice.open(0);

    // namedWindow("Output", 1);

    Mat captureFrame;
    //auto sleepDuration = std::chrono::duration <double, std::milli>(20).count();
    //std::thread running(run);
    //SVMClassifier svmClassifier(PEOPLE_NUMBER, NUMBER_OF_PICTURES_FROM_EVERY_PERSON, FEATURE_DIMENSION);
    //svmClassifier.setTraningData(dataBaseFeatures);
    //svmClassifier.setTraningParameters();
    //auto start = chrono::steady_clock::now();

    //svmClassifier.train();

    //auto end = chrono::steady_clock::now();

    //cout << "Elapsed time of training in milliseconds : "
    //    << chrono::duration_cast<chrono::milliseconds>(end - start).count()
    //    << " ms" << endl;

    //svmClassifier.saveClassifier("C:\\Users\\Elen\\Desktop\\trainFile.xml");
    //std::cout << "SVM is trained\n";
    
    while (true) {
        //capture a new image frame
        captureDevice >> captureFrame;
        bool newImage = getFacesFromVideo(captureFrame, face_cascade);
        
        //if new face was loaded, get features from it
        if (newImage) {
            auto input_features = getFeature(SRC_DIR);
            auto start = chrono::steady_clock::now();
            std::cout << getLabel(&(input_features[0][0]), svm);
            std::string name = getName(getLabel(&(input_features[0][0]), svm));
            auto end = chrono::steady_clock::now();
            cout << "Elapsed time of prediction in milliseconds : "
                << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
                << " ns" << endl;
            if (name != "") {
                std::cout << "This is " + name << ".\n";
                if (true) {
                    cv::imshow(name, captureFrame);
                    waitKey(3);
                }
            }
        }
       
        
     }

    return 0;
}


