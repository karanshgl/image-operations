#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include<string>
#include <cmath>

#define PI 3.14159265358979323846

using namespace cv;
using namespace std;

enum Interpolation { nearest, bilinear };

class AffineTransformation{

  int bilinearInterpolation(Mat &I, int r, int c, double dr, double dc, int k){
    
    

    double val = I.at<Vec3b>(r,c).val[k]*(1-dr)*(1-dc);
    val += I.at<Vec3b>(r+1,c).val[k]*(dr)*(1-dc);
    val += I.at<Vec3b>(r,c+1).val[k]*(1-dr)*(dc);
    val += I.at<Vec3b>(r+1,c+1).val[k]*(dr)*dc;

    int intensity = round(val);

    return intensity;
  }

  int nearestNeighbour(Mat &I, int x, int y, int k){

    return I.at<Vec3b>(x,y).val[k];
  }

  double degree_to_radian(double degree){
    return degree*PI/180;
  }

  double radian_to_degree(double radian){
    return radian*180/PI;
  }

  bool inRange(int x, int be, int en){
    return (be <= x && x < en);
  }

  Mat resample(Mat &I, int newRow, int newCol, Interpolation ip){

    int nRows = I.rows;
    int nCols = I.cols;
    int channels = I.channels();

    Mat output(newRow, newCol, CV_8UC3);

    double sr = (nRows-1)*1.0/(newRow-1);
    double sc = (nCols-1)*1.0/(newCol-1);

    Vec3b *p,*q;

    for(int i=0;i<newRow;i++){
      p = I.ptr<Vec3b>(i);
      q = output.ptr<Vec3b>(i);
      for(int j=0;j<newCol;j++){
        

        for(int k=0;k<channels;k++){
          if(ip == nearest){

            int x = round(i*sr);
            int y = round(j*sc);

            if(inRange(x,0,nRows) && inRange(y, 0, nCols)) q[j][k] = nearestNeighbour(I, x, y ,k);
          } 

          else if(ip == bilinear){

            int r = floor(i*sr);
            int c = floor(j*sc);

            double dr = i*sr - r;
            double dc = j*sc - c;

            if(inRange(r,0,nRows) && inRange(c, 0, nCols)) q[j][k] = bilinearInterpolation(I, r, c, dr, dc, k);
          } 
        }
      }
    }
    return output;
  }

public:

  Mat resize(Mat &I, double scale, Interpolation ip){

    int nRows = I.rows;
    int nCols = I.cols;

    int newRow = floor(scale*nRows);
    int newCol = floor(scale*nCols);

    return resample(I, newRow, newCol, ip);

  }

Mat rotate(Mat &I, double angle, Interpolation ip){

  int channels = I.channels();
  int nRows = I.rows;
  int nCols = I.cols;

  double radian = degree_to_radian(angle);
  double sine = sin(radian);
  double cosine = cos(radian);

  if(angle>=180) angle-=360;


  double diagonal = sqrt(nRows*nRows + nCols*nCols);
  double theta = radian_to_degree(atan(nRows*1.0/nCols));

  int newRow = 0;
  int newCol = 0;
  double theta_x = 0;
  double theta_y = 0;

  if(0 <= angle && angle < 90){
    // First Quad
    theta_x = degree_to_radian(angle + theta);
    theta_y = degree_to_radian(angle - theta);
    newRow = round(abs(diagonal*sin(theta_x)));
    newCol = round(abs(diagonal*cos(theta_y)));
  }

  else if(-90 <= angle && angle < 0){
    // Fourth Quad
    theta_x = degree_to_radian(angle - theta);
    theta_y = degree_to_radian(angle + theta);

    newRow = round(abs(diagonal*sin(theta_x)));
    newCol = round(abs(diagonal*cos(theta_y)));
  }

  else if(90 <= angle && angle < 180){
    // Second
    theta_x = degree_to_radian(angle - theta - 90);
    theta_y = degree_to_radian(angle + theta - 90);

    newRow = round(abs(diagonal*cos(theta_x)));
    newCol = round(abs(diagonal*sin(theta_y)));
  }

  else {
    theta_x = degree_to_radian(angle + theta + 90);
    theta_y = degree_to_radian(angle - theta + 90);

    newRow = round(abs(diagonal*cos(theta_x)));
    newCol = round(abs(diagonal*sin(theta_y)));
  }

  cout<<newRow<<" "<<newCol<<endl;

  int in_origin_x = floor(0.5*nRows);
  int in_origin_y = floor(0.5*nCols);

  int out_orgin_x = floor(0.5*newRow);
  int out_orgin_y = floor(0.5*newCol);

  Mat output(newRow, newCol, CV_8UC3);

  Vec3b *p,*q;

  for(int i=0;i<newRow;i++){
    
    int shifted_i = i - out_orgin_x;

    p = I.ptr<Vec3b>(i);
    q = output.ptr<Vec3b>(i);

    for(int j=0;j<newCol;j++){

      int shifted_j = j - out_orgin_y;

      double rotate_x = cosine*shifted_i - sine*shifted_j + in_origin_x;
      double rotate_y = sine*shifted_i + cosine*shifted_j + in_origin_y;

      for(int k=0;k<channels;k++){
        if(ip == nearest){
            int r = round(rotate_x);
            int c = round(rotate_y);

            if(inRange(r,0,nRows) && inRange(c, 0, nCols)) q[j][k] = nearestNeighbour(I, r, c ,k);
        } 

        else if(ip == bilinear){

          int r = floor(rotate_x);
          int c = floor(rotate_y);

          double dr = rotate_x - r;
          double dc = rotate_y - c;

          if(inRange(r,0,nRows) && inRange(c, 0, nCols)) q[j][k] = bilinearInterpolation(I, r, c, dr, dc, k);
        } 
      }

    }
  }
  return output;

}

Mat scale(Mat &I, double x_scale, double y_scale, Interpolation ip){

  int nRows = I.rows;
  int nCols = I.cols;

  int newRow = floor(x_scale*nRows);
  int newCol = floor(y_scale*nCols);

  return resample(I, newRow, newCol, ip);
}

Mat translate(Mat &I, int dx, int dy){

  int Rows = I.rows;
  int Cols = I.cols;

  Mat output = Mat::zeros(Rows, Cols, CV_8UC3);

  for(int i=0;i<Rows;i++){
  
    for(int j=0;j<Cols;j++){
      if(i+dx < Rows && i+dx >=0 && j+dy < Cols && j+dy >= 0){
        output.at<Vec3b>(i+dx,j+dy) = I.at<Vec3b>(i,j);
      }
    }
  }
  return output;
}

Mat shear(Mat &I, double factor, char axis, Interpolation ip ){

  int nRows = I.rows;
  int nCols = I.cols;
  int channels = I.channels();

  double shear_x = 0, shear_y = 0;

  if(axis == 'x') shear_x = factor;
  else if(axis == 'y') shear_y = factor;

  int newRow = floor(nRows + shear_x*nCols);
  int newCol = floor(nCols + shear_y*nRows);


  Mat output(newRow, newCol, CV_8UC3);

  Vec3b *p,*q;

  for(int i=0;i<newRow;i++){
    
    p = I.ptr<Vec3b>(i);
    q = output.ptr<Vec3b>(i);

    for(int j=0;j<newCol;j++){

      for(int k=0;k<channels;k++){
        if(ip == nearest){
            int r = round((i - shear_x*j)/(1-shear_x*shear_y));
            int c = round((j - shear_y*i)/(1-shear_x*shear_y));
            if(inRange(r,0,nRows) && inRange(c, 0, nCols)) q[j][k] = nearestNeighbour(I, r, c ,k);
        } 

        else if(ip == bilinear){

          int r = floor((i - shear_x*j)/(1-shear_x*shear_y));
          int c = floor((j - shear_y*i)/(1-shear_x*shear_y));

          double dr = (i - shear_x*j)/(1-shear_x*shear_y) - r;
          double dc = (j - shear_y*i)/(1-shear_x*shear_y) - c;

          if(inRange(r,0,nRows) && inRange(c, 0, nCols)) q[j][k] = bilinearInterpolation(I, r, c, dr, dc, k);
        } 
      }

    }
  }
  return output;
}

};


class IntensityTransformations{
  double lookuptable_log[256];
public:

  IntensityTransformations(){

    for(int i=0;i<256;i++){
      lookuptable_log[i] = log(1+i);
      cout<<lookuptable_log[i]<< " ";
    }
  }

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

  Mat negative_gray(Mat &I){
    // Returns the negative of the image 
    int nRows = I.rows;
    int nCols = I.cols;

    Mat I_gray(I);

    cvtColor(I, I_gray, CV_BGR2GRAY );

    Mat output(nRows,nCols,CV_8UC1);

    uchar *p,*q;

    for(int i=0;i<nRows;i++){
      p = I_gray.ptr<uchar>(i);
      q = output.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){
        q[j] = 255 - p[j];
      }
    }
    return output;
  }

  Mat logTransformation(Mat &I, double c = 1.0){

    int nRows = I.rows;
    int nCols = I.cols;
    int channels = I.channels();

    Mat I_LAB;
    cvtColor(I, I_LAB, COLOR_BGR2HSV);

    Vec3b *p;

    for(int i=0;i<nRows;i++){
      p = I_LAB.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        int intensity = round(c*lookuptable_log[p[j][2]]);
        p[j][2] = (intensity > 255 ? 255 : intensity );
      }
    }
    cvtColor(I_LAB, I_LAB, CV_HSV2BGR);
    return I_LAB;


  }

  Mat logTransformation_gray(Mat &I, double c = 1.0){

    int nRows = I.rows;
    int nCols = I.cols;

    Mat I_gray(I);

    cvtColor(I, I_gray, CV_BGR2GRAY );

    Mat output(nRows,nCols,CV_8UC1);

    uchar *p,*q;

    for(int i=0;i<nRows;i++){
      p = I_gray.ptr<uchar>(i);
      q = output.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){
        int intensity = round(c*lookuptable_log[p[j]]);
        q[j] = (intensity > 255 ? 255 : intensity );
      }
    }
    return output;
  }

  Mat gammaCorrection_gray(Mat &I,  double gamma = 1.0, double c = 1.0){

    int nRows = I.rows;
    int nCols = I.cols;

    Mat I_gray(I);

    cvtColor(I, I_gray, CV_BGR2GRAY );

    Mat output(nRows,nCols,CV_8UC1);

    uchar *p,*q;

    for(int i=0;i<nRows;i++){
      p = I_gray.ptr<uchar>(i);
      q = output.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){
        double base = p[j]*1.0/255;
        int intensity = round(c*pow(base,1/gamma)*255);
        q[j] = (intensity > 255 ? 255 : intensity );
      }
    }
    return output;
  }

};


int main( int argc, char** argv ) {
  
  Mat image, img;
  image = imread("ein.jpg" , 1);
  
  if(! image.data ) {
      cout <<  "Could not open or find the image" << endl ;
      return -1;
    }
  IntensityTransformations a;
  image = a.gammaCorrection_gray(image, 0.5);
  namedWindow( "Display window", WINDOW_AUTOSIZE );
  imshow( "Display window", image );
  
  waitKey(0);
  return 0;
}
