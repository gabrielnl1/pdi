#include <QApplication>

#include <iostream>
#include <string>

#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    cv::Mat img = cv::imread("budismo-buda.jpg", IMREAD_GRAYSCALE);
    namedWindow("img", WINDOW_KEEPRATIO);
    imshow("img", img);
    return a.exec();
}
