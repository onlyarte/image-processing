#include <cmath>
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

int main()
{
    string path ("D:/DEV/Project/berlin.jpeg");
    Mat src;

	src = imread(path.c_str(), IMREAD_GRAYSCALE );

    Mat g = gauss(src);

    imshow("Source image", src);
    imshow("Gauss", g);

    waitKey(0);

    return 0;
}
