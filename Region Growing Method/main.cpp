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


/*region growing method
start_x, start_y -- point coordinates, from which growing begins
*/
Mat regionGrowing(const Mat &src, const unsigned int start_x = 0, const unsigned int start_y = 0, const unsigned int accur = 10){

    //number of cols and rows
    int nc = src.cols, nr = src.rows;

    //create new image
    Mat res(Size(nc, nr), CV_8UC1);

    //check input
    if (src.data == 0 || src.rows < 1 || src.cols < 1 || start_y*nc + start_x >= nr*nc){
        cout << "Wrong data" << endl;
        return res;
    }

    const unsigned int s_value = src.data[start_y*nc + start_x];
    int min = s_value - accur,
        max = s_value + accur;
    if(min < 0) min = 0;
    if(max > 255) max = 255;
    //indicate background color
    int back_color = 0;
    if(s_value < 128)
        back_color = 255;
    //fill background
    for(int i = 0; i < nc*nr; i++)
        res.data[i] = back_color;

        cout << s_value << endl;
        cout << min << " " << max << endl;

    //stack for points to check
    vector<int> stack;
    stack.push_back(start_x*nc + start_y);

    while(stack.size()>0){
        int curr = stack[0];//get last from stack
        stack.erase(stack.begin());//remove it

        //if already added skip
        if(res.data[curr] != back_color)
            continue;
        //else add
        res.data[curr] = src.data[curr];

        //check 8 neighbours
        int nbs[8]; // neighbours
            nbs[0] = curr - nc;     // nn
            nbs[1] = curr + nc;     // ss
            nbs[2] = curr + 1;      // ww
            nbs[3] = curr - 1;      // ee
            nbs[4] = nbs[0] + 1;    // nw
            nbs[5] = nbs[0] - 1;    // ne
            nbs[6] = nbs[1] + 1;    // sw
            nbs[7] = nbs[1] - 1;    // se

        //add similar neighbours to stack
        for(int i = 0; i < 8; i++)
            if(nbs[i] >= 0 && nbs[i] < nc*nr &&
               src.data[nbs[i]] >= min && src.data[nbs[i]] <= max)
                stack.push_back(nbs[i]);
    }

    return res;
}

int main()
{
    string path ("D:/DEV/Project/icon.png");
    Mat src;

    src = imread(path.c_str(), IMREAD_GRAYSCALE );

    Mat s = regionGrowing(src, 100, 100, 40);

    imshow("Source image", src);
    imshow("Region", s);

    waitKey(0);

    return 0;
}
