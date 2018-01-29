#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class AffineTransformation{
public:

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
  Levels l;
  image = l.negative(image);
  namedWindow( "Display window", WINDOW_AUTOSIZE );
  imshow( "Display window", image );
  
  waitKey(0);
  return 0;
}
