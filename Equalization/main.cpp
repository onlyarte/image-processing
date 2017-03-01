#include <iostream>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

/*

Author: Ruslan Purii
Group 5

*/

int getNewValue(const int &value, vector<int> &uniq, vector<int> &cdf){
    for(int i = 0; i <  uniq.size(); i++){
        if(uniq[i] == value){
            return cdf[i];
        }
    }
    return 255;
}

Mat equalize(const Mat &src){
    //check input
    if (src.data == 0 || src.rows < 1 || src.cols < 1)
        cout << "Could not open or find the image" << endl;

    //create copy to change
    Mat res(Size(src.cols, src.rows), CV_8UC1);

    int length = src.rows * src.cols;

    //copy data
    vector<int> data(length);

    for(int i = 0; i <  length; i++)
        data[i] = src.data[i];

    //sort data
    sort(data.begin(), data.end());

    //make data values unique, count frequency
    vector<int> uniq, freq;
    for(int i = 0, prev = -1, counter = 0; i < length; i++){
        if(data[i] != prev){
            prev = data[i];
            uniq.push_back(prev);
            freq.push_back(1);
            counter++;
        }
        else
            freq[counter-1]++;
    }

    //solve pmf, cdf[0;1]
    vector<double> precdf(uniq.size());
    double prevSum = 0.0;
    cout << prevSum<< endl;
    for(int i = 0; i < uniq.size(); i++){
        precdf[i] = (double)freq[i]/length + prevSum;
        prevSum = precdf[i];
    }

    //solve cdf[0;255]
    vector<int> cdf(uniq.size());
    for(int i = 0; i < uniq.size(); i++){
        cdf[i] = (double)round(precdf[i] * 255);
    }

    //remap
    for(int i = 0; i < length; i++){
        res.data[i] = getNewValue(src.data[i], uniq, cdf);
    }

    return res;
}

Mat equalizeOpenCV(const Mat &src){
    //check input
    if (src.data == 0 || src.rows < 1 || src.cols < 1)
        cout << "Could not open or find the image" << endl;
    //create copy to change
    Mat res(Size(src.cols, src.rows), CV_8UC1);

    equalizeHist( src, res );

    return res;
}

Mat showHist(const Mat &src){
    Mat dst;

  if( !src.data )
    cout << "Could not open or find the image" << endl;

  //get values
  vector<Mat> data;
  split( src, data );

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat hist;

  /// Compute the histograms:
  calcHist( &data[0], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h + 20, hist_w + 20, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       Scalar( 255, 255, 255), 2, 8, 0  );
  }

  return histImage;
}

int main()
{
    string path ("D:/DEV/Project/Geneva.tif");
    Mat src;

	src = imread(path.c_str(), IMREAD_GRAYSCALE );

    Mat equalized = equalize(src);
    Mat openCV = equalizeOpenCV(src);
    Mat histOriginal = showHist(src);
    Mat histEqualized = showHist(equalized);

    imshow("Source image", src);
    imshow("Normalized image", equalized);
    imshow("OpenCV", openCV);
    imshow("Hist Original", histOriginal);
    imshow("Hist Equalized", histEqualized);

    waitKey(0);

    return 0;
}
