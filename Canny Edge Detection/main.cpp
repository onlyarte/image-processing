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

//return x-gradient value after applying sobel x-matrix to neighbour pixels
int xGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y, x-1) +
                 image.at<uchar>(y+1, x-1) -
                  image.at<uchar>(y-1, x+1) -
                   2*image.at<uchar>(y, x+1) -
                    image.at<uchar>(y+1, x+1);
}

//return y-gradient value after applying sobel y-matrix to neighbour pixels
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
Mat canny(const Mat src){
    //check input
    if (src.data == 0){
        cout << "Wrong data" << endl;
        return src;
    }
    //number of cols/rows
    unsigned int nc = src.cols,
        nr = src.rows;
    //create new images
    Mat prev(Size(nc, nr), CV_8UC1),
        next(Size(nc, nr), CV_8UC1);

    //copy data to start image
    for(int i = 0; i < nc*nr; i++)
        prev.data[i] = src.data[i];

    // 1. APPLY GAUSS FILTER
    prev = gauss(prev);

    // 2. APPLY SOBEL FILTER
    //save gradients to vector
    vector<int> gxs(nc*nr);
    vector<int> gys(nc*nr);
    //for each pixel find gradients and new values
    double gx, gy, sum;
    for(int y = 1; y < nr - 1; y++){
        for(int x = 1; x < nc - 1; x++){
            gx = xGradient(prev, x, y);
            gy = yGradient(prev, x, y);
            sum = abs(gx) + abs(gy);
            //if out of bounds
            sum = sum > 255 ? 255 : sum;
            sum = sum < 0 ? 0 : sum;
            //rewrite pixel value
            next.at<uchar>(y,x) = sum;
            //save for non-maximum suppression
            gxs.push_back(abs(gx)); gys.push_back(abs(gy));
        }
    }
    prev = next;

    // 3. NON-MAXIMUM SUPPRESSION
    for (int i = 1; i < nr - 1; i++){
        for (int j = 1; j < nc - 1; j++){
            //find current location of neighbour pixels: nn - north, ne - north east
            const int c = i + nr * j,
                nn = c - nr, ss = c + nr,
                ww = c + 1, ee = c - 1,
                nw = nn + 1, ne = nn - 1,
                sw = ss + 1, se = ss - 1;
            //indicate direction using sobel gradient and rounding it to fit 1-8 measures
            const float dir = (float)(fmod(atan2(gys[c],
                                                 gxs[c]) + M_PI,
                                           M_PI) / M_PI) * 8;
            //if non-maximum then remove
            if (!(((dir <= 1 || dir > 7) && prev.data[c] > prev.data[ee] &&
                 prev.data[c] > prev.data[ww]) || // 0 deg
                ((dir > 1 && dir <= 3) && prev.data[c] > prev.data[nw] &&
                 prev.data[c] > prev.data[se]) || // 45 deg
                ((dir > 3 && dir <= 5) && prev.data[c] > prev.data[nn] &&
                 prev.data[c] > prev.data[ss]) || // 90 deg
                ((dir > 5 && dir <= 7) && prev.data[c] > prev.data[ne] &&
                 prev.data[c] > prev.data[sw])))   // 135 deg
                next.data[c] = 0;
        }
    }
    prev = next;
    next = Mat(Size(nc, nr), CV_8UC1);

    //trace edges with hysteresis
    vector<int> edges(nc*nr/2);
    const int tmin = 150, tmax = 200;
    int c = 1;
    for (int j = 1; j < nc - 1; j++)
        for (int i = 1; i < nr - 1; i++) {
            if (prev.data[c] >= tmax && next.data[c] == 0) { // trace edges
                next.data[c] = 255;
                int nedges = 1;
                edges[0] = c;

                do {
                    nedges--;
                    const int t = edges[nedges];

                    int nbs[8]; // neighbours
                    nbs[0] = t - nr;     // nn
                    nbs[1] = t + nr;     // ss
                    nbs[2] = t + 1;      // ww
                    nbs[3] = t - 1;      // ee
                    nbs[4] = nbs[0] + 1; // nw
                    nbs[5] = nbs[0] - 1; // ne
                    nbs[6] = nbs[1] + 1; // sw
                    nbs[7] = nbs[1] - 1; // se
                    //join neighbour if its value more then min and not joined yet
                    for (int k = 0; k < 8; k++)
                        if (prev.data[nbs[k]] >= tmin && next.data[nbs[k]] == 0) {
                            next.data[nbs[k]] = 255;
                            edges[nedges] = nbs[k];
                            nedges++;
                        }
                } while (nedges > 0);
            }
            c++;
        }
    return next;
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
