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


//gauss filter
Mat gauss(const Mat &src){

    //create new image
    Mat res(Size(src.cols, src.rows), CV_8UC1);

    //check input
    if (src.data == 0 || src.rows < 1 || src.cols < 1){
        cout << "Wrong data" << endl;
        return res;
    }

    int mask[] = {2, 4, 5, 4, 2,
        4, 9, 12, 9, 4,
        5, 12, 15, 12, 5,
        4, 9, 12, 9, 4,
        2, 4, 5, 4, 2};

    const int size = 5;
    const int coeff = 159;
    const int m = size, n = size;
    const int a = (m-1)/2, b = (n-1)/2;
    //for each col of each row
    for(int i = 0; i < res.rows; i++){
        for(int j = 0; j < res.cols; j++){
            //get new pixel value sum
            double sum = 0;
            //find sum adding and multiplying src data to matrix
            for(int k = 0, x = i - a; k < m; k++, x++){
                for(int l = 0, z = j - b; l < n; l++, z++){
                    //if no such pixel: (-1, -1) or so
                    if(x < 0 || z < 0 || x >= res.rows || z >= res.cols)
                        sum += src.at<uchar>(0,0) * mask[k*size + l];
                    else
                        sum += src.at<uchar>(x,z) * mask[k*size + l];
                }
            }
            //write new data
            res.at<uchar>(i,j) = (int)sum/coeff;
        }
    }
    return res;
}

//for sobel filter
int xGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y, x-1) +
                 image.at<uchar>(y+1, x-1) -
                  image.at<uchar>(y-1, x+1) -
                   2*image.at<uchar>(y, x+1) -
                    image.at<uchar>(y+1, x+1);
}

//for sobel filter
int yGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y-1, x) +
                 image.at<uchar>(y-1, x+1) -
                  image.at<uchar>(y+1, x-1) -
                   2*image.at<uchar>(y+1, x) -
                    image.at<uchar>(y+1, x+1);
}

//Canny edge detection
Mat canny(Mat src){
    //create new image
    Mat res(Size(src.cols, src.rows), CV_8UC1);

    //check input
    if (src.data == 0){
        cout << "Wrong data" << endl;
        return res;
    }

    // 1. APPLY GAUSS FILTER
    src = gauss(src);

    // 2. APPLY SOBEL FILTER
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
            gxs.push_back(abs(gx)); gys.push_back(abs(gy));
        }
    }

    // 3. NON-MAXIMUM SUPPRESSION
    for (int i = 1; i < res.rows - 1; i++){
        for (int j = 1; j < res.cols - 1; j++){
            const int c = i + res.rows * j,
                nn = c - res.rows, ss = c + res.rows,
                ww = c + 1, ee = c - 1,
                nw = nn + 1, ne = nn - 1,
                sw = ss + 1, se = ss - 1;

            const float dir = (float)(fmod(atan2(gys[c],
                                                 gxs[c]) + M_PI,
                                           M_PI) / M_PI) * 8;

            if (!(((dir <= 1 || dir > 7) && res.data[c] > res.data[ee] &&
                 res.data[c] > res.data[ww]) || // 0 deg
                ((dir > 1 && dir <= 3) && res.data[c] > res.data[nw] &&
                 res.data[c] > res.data[se]) || // 45 deg
                ((dir > 3 && dir <= 5) && res.data[c] > res.data[nn] &&
                 res.data[c] > res.data[ss]) || // 90 deg
                ((dir > 5 && dir <= 7) && res.data[c] > res.data[ne] &&
                 res.data[c] > res.data[sw])))   // 135 deg
                res.data[c] = 0;
        }
    }
    return res;
}

int main()
{
    string path ("D:/DEV/Project/Geneva.tif");
    Mat src;

    src = imread(path.c_str(), IMREAD_GRAYSCALE );

    Mat s = canny(src);

    imshow("Source image", src);
    imshow("Mask", s);

    waitKey(0);

    return 0;
}
