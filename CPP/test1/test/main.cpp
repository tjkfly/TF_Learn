#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace std;
using namespace cv::dnn;
int main()
{
//    Mat frame = imread("/home/tjk/project/python/11.png");
//    imshow("0",frame);
//    cvtColor(frame,frame,COLOR_BGR2GRAY);
//    resize(frame,frame,Size(28,28));
//    imshow("1",frame);
//    imwrite("/home/tjk/project/python/11.png",frame);
//    waitKey(0);
    string path = "/home/tjk/project/python/model_save/model/saved_model.pb";
    Net net = readNetFromTensorflow(path);
    printf("第一步");

}
