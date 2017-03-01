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


//sobel y gradient
int xGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y, x-1) +
                 image.at<uchar>(y+1, x-1) -
                  image.at<uchar>(y-1, x+1) -
                   2*image.at<uchar>(y, x+1) -
                    image.at<uchar>(y+1, x+1);
}

//sobel y gradient
int yGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y-1, x) +
                 image.at<uchar>(y-1, x+1) -
                  image.at<uchar>(y+1, x-1) -
                   2*image.at<uchar>(y+1, x) -
                    image.at<uchar>(y+1, x+1);
}


//return sobel filtered image
Mat sobel(const Mat &src){
    //create new image
    Mat res(Size(src.cols, src.rows), CV_8UC1);

    vector<double> vec(src.cols*src.rows);

    //check input
    if (src.data == 0){
        cout << "Wrong data" << endl;
        return res;
    }

    //save gradients to vector
    vector<int> gxs(res.rows*res.cols);
    vector<int> gys(res.rows*res.cols);
    //for each pixel find gradients and new values
    double gx, gy, sum;
    for(int y = 1; y < src.rows - 1; y++){
        for(int x = 1; x < src.cols - 1; x++){
            gx = xGradient(src, x, y);
            gy = yGradient(src, x, y);
            sum = abs(gx) + abs(gy);
            sum = sum > 255 ? 255:sum;
            sum = sum < 0 ? 0 : sum;
            res.at<uchar>(y,x) = sum;
        }
    }

    return res;
}

int main()
{
    string path ("D:/DEV/Project/berlin.jpeg");
    Mat src;

    src = imread(path.c_str(), IMREAD_GRAYSCALE );

    Mat g = sobel(src);

    imshow("Source image", src);
    imshow("Sobel", g);

    waitKey(0);

    return 0;
}
