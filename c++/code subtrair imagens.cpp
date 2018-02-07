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
 
    // Subtrair duas imagens e pegar o módulo para exibir na tela
    
    int sum = 0;
    cv::Mat img = cv::imread("lena.png", IMREAD_COLOR);
    cv::Mat img2 = cv::imread("512buda.png", IMREAD_COLOR);
    Mat bin;
    namedWindow("bin", WINDOW_KEEPRATIO);
    bin = abs(img - img2);
    imshow("bin", bin);
    


    return a.exec();
}

