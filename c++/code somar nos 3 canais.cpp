#include <QApplication>

#include <iostream>
#include <string>

#include <cv.h>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
 
       //Somar uma imagem com um valor escalar nos três canais (RGB)
    
    int sum = 127;
    cv::Mat img = cv::imread("budismo-buda.jpg", IMREAD_COLOR);
    Mat bin;
    namedWindow("bin", WINDOW_KEEPRATIO);
    createTrackbar("sum", "bin", &sum, 255, 0);
    for (;;)
    {
        bin = img + Scalar(sum,sum,sum);
        imshow("bin", bin);
        if ((char) waitKey(1) == 'q') break;
    }
    


    return a.exec();
}

