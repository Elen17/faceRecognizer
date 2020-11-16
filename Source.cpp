#include<iostream>
using namespace std;
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"
#include "math_functions.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace seeta;

#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "

std::string DATA_DIR = "C:\\Users\\Elen\\Downloads\\SeetaFaceEngine-master\\FaceIdentification\\data\\";
std::string MODEL_DIR = "C:\\Users\\Elen\\Downloads\\SeetaFaceEngine-master\\FaceIdentification\\model\\";

int main(int argc, char* argv[]) {
	// Initialize face detection model
	cout << "STARTED!";
	seeta::FaceDetection detector((MODEL_DIR + "seeta_fd_frontal_v1.0.bin").c_str());
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model 
	seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());

	// Initialize face Identification model 
	FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
	std::string test_dir = DATA_DIR + "test_face_recognizer\\";

	cout << "LOADING IMAGE...";


	//load image
	cv::Mat gallery_img_color = cv::imread("C:\\Users\\Elen\\Downloads\\SeetaFaceEngine-master\\FaceIdentification\\data\\test_face_recognizer\\images\\compare_im\\Aaron_Peirsol_0001.jpg", 1);
	cv::Mat gallery_img_gray = cv::imread("C:\\Users\\Elen\\Downloads\\SeetaFaceEngine-master\\FaceIdentification\\data\\test_face_recognizer\\images\\compare_im\\Aaron_Peirsol_0001.jpg", 0);
	//cv::cvtColor(gallery_img_color, gallery_img_gray, cv::COLOR_BGR2GRAY);
	cv::namedWindow("Test1", cv::WINDOW_AUTOSIZE);
	cv::imshow("Test1", gallery_img_color);
	cv::waitKey(0);
	cv::Mat probe_img_color = cv::imread("C:\\Users\\Elen\\source\\repos\\Project5\\Project5\\data\\images\\emmaWatson\\image1.jpg", 1);
	cv::Mat probe_img_gray = cv::imread("C:\\Users\\Elen\\source\\repos\\Project5\\Project5\\data\\images\\emmaWatson\\image1.jpg", 0);
	cv::namedWindow("Test2", cv::WINDOW_AUTOSIZE);
	cv::imshow("Test2", probe_img_color);
	cv::waitKey(0);
	//cv::cvtColor(probe_img_color, probe_img_gray, cv::COLOR_BGR2GRAY);

	ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
	gallery_img_data_color.data = gallery_img_color.data;

	ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
	gallery_img_data_gray.data = gallery_img_gray.data;

	ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
	probe_img_data_color.data = probe_img_color.data;

	ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
	probe_img_data_gray.data = probe_img_gray.data;

	// Detect faces
	std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
	int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());

	std::vector<seeta::FaceInfo> probe_faces = detector.Detect(probe_img_data_gray);
	int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());

	if (gallery_face_num == 0 || probe_face_num == 0)
	{
		std::cout << "Faces are not detected.";
		return 0;
	}

	// Detect 5 facial landmarks
	seeta::FacialLandmark gallery_points[5];
	point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);

	seeta::FacialLandmark probe_points[5];
	point_detector.PointDetectLandmarks(probe_img_data_gray, probe_faces[0], probe_points);

	for (int i = 0; i < 5; i++)
	{
		cv::circle(gallery_img_color, cv::Point(gallery_points[i].x, gallery_points[i].y), 2,
			cv::Scalar(0, 255, 0));
		cv::circle(probe_img_color, cv::Point(probe_points[i].x, probe_points[i].y), 2,
			cv::Scalar(0, 255, 0));
	}
	cv::imwrite("gallery_point_result.jpg", gallery_img_color);
	cv::imwrite("probe_point_result.jpg", probe_img_color);

	// Extract face identity feature
	float gallery_fea[2048];
	float probe_fea[2048];
	face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
	face_recognizer.ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);

	// Caculate similarity of two faces
	float sim = face_recognizer.CalcSimilarity(gallery_fea, probe_fea);
	std::cout << sim << endl;
	cv::waitKey(10);
	system("pause");

	return 0;
}


