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
    int thresh = 127;
    cv::Mat img = cv::imread("budismo-buda.jpg", IMREAD_GRAYSCALE);
    Mat bin;
    namedWindow("bin", WINDOW_KEEPRATIO);
    createTrackbar("thresh", "bin", &thresh, 255, 0);
    for (;;)
    {
        bin = img > thresh;
        imshow("bin", bin);
        if ((char) waitKey(1) == 'q') break;
    }
    return a.exec();
}
