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
 
     // Utilizar o filtro mediana
     
     Mat src = imread( "budismo-buda.jpg", IMREAD_GRAYSCALE );
     Mat dst;

     //Apply median filter
     medianBlur ( src, dst, 7 );
     imshow("source", src);
     imshow("result", dst);

     waitKey(0);
     

    return a.exec();
}

