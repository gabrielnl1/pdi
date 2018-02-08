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

     // código para aplicar o filtro da mediana
     /*
     Mat src = imread( "budismo-buda.jpg", IMREAD_GRAYSCALE );
     Mat dst;

     //Apply median filter
     medianBlur ( src, dst, 7 );
     imshow("source", src);
     imshow("result", dst);

     waitKey(0);
     */


    // Código para criar uma trackbar e fazer BITWISE AND na imagem, com um escalar
    /*
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
    */

    // Preencher uma matriz 3x3 com 127 e printar na tela
    /*
    Mat img = 127 * Mat::ones(3,3,CV_8U);
    cout << img << endl;
    */

    /*

    // Preencher uma matriz 500x500 com números aleatórios normalmente distribuídos e somar com outra imagem e calcular o histograma da imagem com números aleatórios

    Mat N = Mat::zeros(640,480,CV_8UC1);
    cv::randn(N, 127, 30);
    imshow("N", N);

    Mat img = cv::imread("budismo-buda.jpg", IMREAD_GRAYSCALE); //a imagem é 640x480
    imshow("img", img);

    Mat bin;
    bin = img + N;
    imshow("bin", bin);

    //calcular o histograma da imagem N
     /// Establish the number of bins
     int histSize = 256;

     /// Set the ranges ( for B,G,R) )
     float range[] = { 0, 256 } ;
     const float* histRange = { range };

     bool uniform = true; bool accumulate = false;

     Mat gray_hist;

     /// Compute the histograms:
     calcHist( &N, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate );

     // Draw the histograms for B, G and R
     int hist_w = 512; int hist_h = 400;
     int bin_w = cvRound( (double) hist_w/histSize );

     Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

     /// Normalize the result to [ 0, histImage.rows ]
     normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

     /// Draw for each channel
     for( int i = 1; i < histSize; i++ )
     {
         line( histImage, Point( bin_w*(i-1), hist_h - cvRound(gray_hist.at<float>(i-1)) ) ,
                          Point( bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i)) ),
                          Scalar( 255, 255, 255), 2, 8, 0  );

     }

     /// Display
     namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
     imshow("calcHist Demo", histImage );

     waitKey(0);
     */


    /*
    Mat N = Mat::ones(500,500,CV_8UC1);
    N = N*255;
    for (int i =0; i<256;i++) {
        for (int j=0; j<256;j++)
        {
            N.at<int>(i,j) = (1/2*3.14)*exp(-[i-
                    ]);
        }
    }


    imshow("N", N);
    waitKey(0);
    */

    // Subtrair duas imagens e pegar o módulo para exibir na tela
    /*
    int sum = 0;
    cv::Mat img = cv::imread("lena.png", IMREAD_COLOR);
    cv::Mat img2 = cv::imread("512buda.png", IMREAD_COLOR);
    Mat bin;
    namedWindow("bin", WINDOW_KEEPRATIO);
    bin = abs(img - img2);
    imshow("bin", bin);
    */

    //Somar uma imagem com um valor escalar nos três canais (RGB)
    /*
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
    */

    //Somar duas imagens utilizando alpha e beta e a função addWeighted
    /*
    int alpha = 0;
    int beta = 0;
    cv::Mat src1 = cv::imread("512buda.png", IMREAD_COLOR);
    cv::Mat src2 = cv::imread("lena.png", IMREAD_COLOR);
    namedWindow("bin", WINDOW_KEEPRATIO);
    createTrackbar("alpha", "bin", &alpha, 255, 0);
    for (;;)
    {
        Mat dst;
        beta = ( 255 - alpha );
        addWeighted( src1, (double)alpha/255, src2, (double)beta/255, 0.0, dst);
        imshow( "bin", dst );
        if ((char) waitKey(1) == 'q') break;
    }
    */

    return a.exec();
}

