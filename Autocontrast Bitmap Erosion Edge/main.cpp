#include <cmath>
#include <iostream>
#include <vector>
#include <stdlib.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

/*

Author: Ruslan Purii
Group 5

*/

//autoregulate contrast
Mat autoContrast(const Mat &src){
    //create new image
    Mat res(Size(src.cols, src.rows), CV_8UC1);

    //check input
    if (src.data == 0){
        cout << "Wrong data" << endl;
        return res;
    }
    //find min/max
    int min = 255, max = 0;
    for(int i = 0; i < src.rows*src.cols; i++){
        if(src.data[i]>max) max = src.data[i];
        if(src.data[i]<min) min = src.data[i];
    }

    //define proper contrast
    for(int i = 0; i < src.rows*src.cols; i++)
        res.data[i] = 255 * (src.data[i] - min) / (max - min);

    return res;
}

//get bitmap image from grayscale
Mat toBitMap(const Mat &src){
    //create new image
    Mat res(Size(src.cols, src.rows), CV_8UC1);

    //check input
    if (src.data == 0){
        cout << "Wrong data" << endl;
        return res;
    }
    //for each pixel value. if more than 127 then 255, else 0
    for(int i = 0; i < src.rows*src.cols; i++){
        if(src.data[i]>127) res.data[i] = 255;
        else res.data[i] = 0;
    }
    return res;
}

//image erosion with 3x3 primitive
Mat erosion(const Mat &src){
    //create new image
    Mat res(Size(src.cols, src.rows), CV_8UC1);

    //check input
    if (src.data == 0){
        cout << "Wrong data" << endl;
        return res;
    }
    //copy data
    for(int i = 0; i < src.rows*src.cols; i++)
        res.data[i] = src.data[i];
    //apply erosion
    for(int i = 1; i < src.rows - 1; i++){
        for(int j = 1; j < src.cols - 1; j++){
            //check square around: if any 0 then 0 else 255
            if(src.at<uchar>(i - 1,j - 1)==0 ||
               src.at<uchar>(i - 1,j)==0 ||
               src.at<uchar>(i - 1,j + 1)==0 ||
               src.at<uchar>(i,j - 1)==0 ||
               src.at<uchar>(i,j + 1)==0 ||
               src.at<uchar>(i + 1,j - 1)==0 ||
               src.at<uchar>(i + 1,j)==0 ||
               src.at<uchar>(i + 1,j + 1)==0)
                res.at<uchar>(i,j) = 0;
            else
                res.at<uchar>(i,j) = 255;
        }
    }

    return res;
}


//find bitmap image edge
Mat edge(const Mat &src){
    //create new image
    Mat res(Size(src.cols, src.rows), CV_8UC1);

    //check input
    if (src.data == 0){
        cout << "Wrong data" << endl;
        return res;
    }
    //copy data to new image
    for(int i = 0; i < src.rows*src.cols; i++)
        res.data[i] = src.data[i];
    //for each pixel
    for(int i = 1; i < src.rows - 1; i++){
        for(int j = 1; j < src.cols - 1; j++){
            //check square around: if not edge then 0 else do nothing
            if(src.at<uchar>(i - 1,j)==src.at<uchar>(i - 1,j - 1) &&
               src.at<uchar>(i - 1,j + 1)==src.at<uchar>(i - 1,j - 1) &&
               src.at<uchar>(i,j - 1)==src.at<uchar>(i - 1,j - 1) &&
               src.at<uchar>(i,j + 1)==src.at<uchar>(i - 1,j - 1) &&
               src.at<uchar>(i + 1,j - 1)==src.at<uchar>(i - 1,j - 1) &&
               src.at<uchar>(i + 1,j)==src.at<uchar>(i - 1,j - 1) &&
               src.at<uchar>(i + 1,j + 1)==src.at<uchar>(i - 1,j - 1))
                res.at<uchar>(i,j) = 0;
            else
                res.at<uchar>(i,j) = 255;
        }
    }
    return res;
}

int main()
{
    string path ("D:/DEV/Project/Geneva.tif");
    Mat src;

    src = imread(path.c_str(), IMREAD_GRAYSCALE );

    Mat s = edge(toBitMap(autoContrast(src)));

    imshow("Source image", src);
    imshow("Out", s);

    waitKey(0);

    return 0;
}
