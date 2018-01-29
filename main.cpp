#include <opencv2/highgui.hpp>
#include <iostream>
#include<string>
#include <cmath>

using namespace cv;
using namespace std;

class AffineTransformation{
public:

  Mat resize(Mat &I, double scale){

    int channels = I.channels();
    int nRows = I.rows;
    int nCols = I.cols;

    int newRow = floor(scale*nRows);
    int newCol = floor(scale*nCols);

    Mat output(newRow, newCol, CV_8UC3);

    double sr = nRows*1.0/newRow;
    double sc = nCols*1.0/newCol;
    cout<<sr<<" "<<sc;
    Vec3b *p,*q;

    for(int i=0;i<newRow;i++){
      p = I.ptr<Vec3b>(i);
      q = output.ptr<Vec3b>(i);
      for(int j=0;j<newCol;j++){
        int r = floor(i*sr);
        int c = floor(j*sc);

        double dr = i*sr - r;
        double dc = j*sc - c;

        for(int k=0;k<channels;k++){
          q[j][k] = bilinear(I,r,c,dr,dc,k);
        }

      }
    }
    return output;

  }

  int bilinear(Mat &I, int r, int c, double dr, double dc, int k){
    double val = I.at<Vec3b>(r,c).val[k]*(1-dr)*(1-dc);
    val += I.at<Vec3b>(r+1,c).val[k]*(dr)*(1-dc);
    val += I.at<Vec3b>(r,c+1).val[k]*(1-dr)*(dc);
    val += I.at<Vec3b>(r+1,c+1).val[k]*(dr)*dc;
    int intensity = round(val);
    return intensity;
  }

};


class Levels{
public:
  Mat negative(Mat &I){
    // Returns the negative of the image 
    int channels = I.channels();
    int nRows = I.rows;
    int nCols = I.cols;

    Mat output(nRows,nCols,CV_8UC3);
    Vec3b *p,*q;

    for(int i=0;i<nRows;i++){
      p = I.ptr<Vec3b>(i);
      q = output.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        for(int k=0;k<channels;k++){
        q[j][k] = 255 - p[j][k];
        }
      }
    }
    return output;
  }
};


int main( int argc, char** argv ) {
  
  Mat image;
  image = imread("la.png" , CV_LOAD_IMAGE_COLOR);
  
  if(! image.data ) {
      cout <<  "Could not open or find the image" << endl ;
      return -1;
    }
  AffineTransformation a;
  image = a.resize(image, 1.5);
  namedWindow( "Display window", WINDOW_AUTOSIZE );
  imshow( "Display window", image );
  
  waitKey(0);
  return 0;
}
