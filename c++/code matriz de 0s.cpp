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
 
    // Preencher uma matriz 3x3 com 127 e printar na tela
    
    Mat img = 127 * Mat::ones(3,3,CV_8U);
    cout << img << endl;
    

    return a.exec();
}

