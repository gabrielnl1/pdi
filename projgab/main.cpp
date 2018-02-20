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
     *
     * ADICIONAR RUIDO DISTRIBUIÇÃO NORMAL MEDIA 127 DESVIO 50 e CALCULAR HISTOGRAMA COLORIDO
     *
     *  Mat src(500, 500, CV_8UC3);
    randn(src, 127, 50);

    Mat dst;

    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split( src, bgr_planes );

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                         Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                         Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                         Scalar( 0, 0, 255), 2, 8, 0  );
    }

    /// Display
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    imshow("calcHist Demo", histImage );
    namedWindow("img original", WINDOW_KEEPRATIO);
    imshow("img original", src );
    waitKey(0);

    // Preencher uma matriz 500x500 com números aleatórios normalmente distribuídos e somar com outra imagem e calcular o histograma da imagem com números aleatórios, tons de cinza

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
    Scalar intensity = img.at<uchar>(y, x);



    Mat N = Mat::ones(500,500,CV_8UC1);
    N = N*255;
    for (int i =0; i<256;i++) {
        for (int j=0; j<256;j++)
        {
            N.at<uchar>(j,i) = (1/2*3.14)*exp(-[i-
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

    // utilizar transformação logaritmica ou exponencial para realçar os detalhes de uma imagem
    /*

    Mat img = imread("spectrum.tif", IMREAD_GRAYSCALE);
    int n = 50;
    namedWindow("img",WINDOW_KEEPRATIO);
    namedWindow("img2",WINDOW_KEEPRATIO);
    createTrackbar("n", "img2", &n, 100, 0,0);
    double c = 1;
    for (;;) {
        //Mat img2 = Mat::zeros(img.rows,img.cols, CV_64F);
        Mat img2;
        img2.create(img.rows, img.cols, CV_64F);
        for (int x = 0; x < img.cols; x++) {
            for (int y = 0; y < img.rows; y++) {
                Scalar intensity = img.at<uchar>(y,x);
                double intensity_new = pow((double)intensity.val[0],25*(double)((n)/1000.0));
                //double intensity_new = c* log (1+(double)intensity.val[0]);
                img2.at<double>(y,x) = intensity_new;
            }
        }
        normalize(img2,img2,1,0,NORM_MINMAX);
        //para normalizar uma imagem que tem valores negativos, a normalização é I-min(I)/max(I)-min(I) ,
        //onde I é a imagem, min(I) é o menor valor dos pixels da imagem, e max(I) é o valor máximo
        img2 = 255*img2;
        img2.convertTo(img2, CV_8U);
        imshow("img", img);
        imshow("img2",img2);
        if((char)waitKey(1) == 'q') break;
    }
    */

    Mat img = imread("kidney.tif", IMREAD_GRAYSCALE);
    int thresh1 = 127;
    int thresh2 = 240;
    namedWindow("img",WINDOW_KEEPRATIO);
    namedWindow("img2",WINDOW_KEEPRATIO);
    createTrackbar("thresh1", "img2", &thresh1, 255, 0,0);
    createTrackbar("thresh2", "img2", &thresh2, 255, 0,0);
    for (;;) {
        //Mat img2 = Mat::zeros(img.rows,img.cols, CV_8U);
        Mat img2;
        img2.create(img.rows, img.cols, CV_8U);
        for (int x = 0; x < img.cols; x++) {
            for (int y = 0; y < img.rows; y++) {
                Scalar intensity = img.at<uchar>(y,x);
                if (intensity.val[0] > thresh1 && intensity.val[0] < thresh2 ) {
                    img2.at<uchar>(y,x) = 255;
                }
                else {
                    img2.at<uchar>(y,x) = 0;
                }
            }
        }
        imshow("img", img);
        imshow("img2",img2);
        if((char)waitKey(1) == 'q') break;
    }

    return a.exec();
}

