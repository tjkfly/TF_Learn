#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace std;
using namespace cv::dnn;
static string class_name[] = {"T-shirt", "Trouser", "Pullover",
                          "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"};
int main()
{
     Mat frame = imread("/home/tjk/project/picture/aj1.png",0);
     imshow("1",frame);
//     cout<<frame.channels()<<endl;
     string path = "/home/tjk/project/tf_doc/frozen_models/frozen_graph.pb";
     Net net = readNetFromTensorflow(path);
     cout<<"模型加载成功"<<endl;
     Mat frame_32F;
     frame.convertTo(frame_32F,CV_32FC1);
//   cout<<1-frame_32F/255.0<<endl;
     Mat blob = blobFromImage(1-frame_32F/255.0,
                              1.0,
                              Size(28,28),
                              Scalar(0,0,0));
//   cout<<(blob.channels());
     net.setInput(blob);
     Mat out = net.forward();
//     cout<<out.cols<<endl;
     Point maxclass;
     minMaxLoc(out, NULL, NULL, NULL, &maxclass);
     cout <<"预测结果为："<<class_name[maxclass.x] << endl;
     waitKey(0);
}
