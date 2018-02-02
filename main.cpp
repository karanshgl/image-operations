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
    
    int row = I.rows;
    int col = I.cols;
    double val = I.at<Vec3b>(r%row,c%col).val[k]*(1-dr)*(1-dc);
    val += I.at<Vec3b>((r+1)%row,c%col).val[k]*(dr)*(1-dc);
    val += I.at<Vec3b>(r%row,(c+1)%col).val[k]*(1-dr)*(dc);
    val += I.at<Vec3b>((r+1)%row,(c+1)%col).val[k]*(dr)*dc;

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

  Mat bilinearConstants(int *arr_x, int *arr_y, int *arr_x_dist, int *arr_y_dist){
    Mat equations(8,8,CV_64FC1, double(0));
    Mat coordinate_distorted(8,1,CV_64FC1);
    Mat c_values(8,1, CV_64FC1);
    double *p;
    int *q;

    for(int i=0;i<8;i++){
      p = equations.ptr<double>(i);
      int offset = (i%2)*4;
      int entry = i/2;
      p[offset] = arr_x[entry];
      p[offset+1] = arr_y[entry];
      p[offset+2] = arr_x[entry]*arr_y[entry];
      p[offset+3] = 1;  
    }

    for(int i=0;i<8;i++){
      p = coordinate_distorted.ptr<double>(i);
      int val = 0;
      if(i%2) val=arr_y_dist[i/2];
      else val=arr_x_dist[i/2];
      p[0] = val;
    }
    Mat eq_inv = equations.inv();
    c_values = eq_inv*coordinate_distorted;

    return c_values;
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

      double rotate_x = cosine*shifted_i + sine*shifted_j + in_origin_x;
      double rotate_y = - sine*shifted_i + cosine*shifted_j + in_origin_y;

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
  double offset_x = 0, offset_y = 0;

  if(axis == 'x'){
    shear_x = factor;
    offset_x = (factor < 0 ? shear_x*nCols: 0 );
  } 
  else if(axis == 'y'){
    shear_y = factor;
    offset_y = (factor < 0 ? shear_y*nRows: 0 );
  } 

  int newRow = floor(nRows + abs(shear_x*nCols));
  int newCol = floor(nCols + abs(shear_y*nRows));


  Mat output(newRow, newCol, CV_8UC3);

  Vec3b *p,*q;

  for(int i=0;i<newRow;i++){
    
    p = I.ptr<Vec3b>(i);
    q = output.ptr<Vec3b>(i);

    for(int j=0;j<newCol;j++){
  
      for(int k=0;k<channels;k++){
        if(ip == nearest){
            int r = round((i - shear_x*j)/(1-shear_x*shear_y) + offset_x);
            int c = round((j - shear_y*i)/(1-shear_x*shear_y) + offset_y);
            if(inRange(r,0,nRows) && inRange(c, 0, nCols)) q[j][k] = nearestNeighbour(I, r, c ,k);
        } 

        else if(ip == bilinear){

          int r = floor((i - shear_x*j)/(1-shear_x*shear_y) + offset_x);
          int c = floor((j - shear_y*i)/(1-shear_x*shear_y) + offset_y);

          double dr = (i - shear_x*j)/(1-shear_x*shear_y) - r + offset_x;
          double dc = (j - shear_y*i)/(1-shear_x*shear_y) - c + offset_y;

          if(inRange(r,0,nRows) && inRange(c, 0, nCols)) q[j][k] = bilinearInterpolation(I, r, c, dr, dc, k);
        } 
      }

    }
  }
  return output;
}

Mat tiePoints(Mat &I, int *arr_x, int *arr_y, int *arr_x_dist, int *arr_y_dist){

  Mat c_orig_dist = bilinearConstants(arr_x, arr_y, arr_x_dist, arr_y_dist);
  Mat c_dist_orig = bilinearConstants(arr_x_dist, arr_y_dist, arr_x, arr_y);

  int nRows = I.rows;
  int nCols = I.cols;

  Vec3b *p, *q;
  double c_d2o[8];
  double c_o2d[8];
  for(int i=0;i<8;i++){
    c_o2d[i] = c_orig_dist.at<double>(i);
    c_d2o[i] = c_dist_orig.at<double>(i);
  }

  int max_x = 0;
  int max_y = 0;
  int min_x = 100000000;
  int min_y = 100000000;
  for(int i=0;i<nRows;i++){
    for(int j=0;j<nCols;j++){

       if(I.at<Vec3b>(i,j).val[0] == 0 && I.at<Vec3b>(i,j).val[1] == 0 && I.at<Vec3b>(i,j).val[2] == 0 ) continue;

       double x_val = c_d2o[0]*i + c_d2o[1]*j + c_d2o[2]*i*j + c_d2o[3];
       double y_val = c_d2o[4]*i + c_d2o[5]*j + c_d2o[6]*i*j + c_d2o[7];
       max_x = max(max_x, (int)round(x_val));
       max_y = max(max_y, (int)round(y_val));
       min_x = min(min_x, (int)round(x_val));
       min_y = min(min_y, (int)round(y_val));
    }
  }

  int dRows = max_x-min_x;
  int dCols = max_y-min_y;

  Mat output(dRows, dCols, CV_8UC3);

  for(int i=0;i<dRows;i++){
    p = output.ptr<Vec3b>(i);
    for(int j=0;j<dCols;j++){
      double x_val = c_o2d[0]*i + c_o2d[1]*j + c_o2d[2]*i*j + c_o2d[3];
      double y_val = c_o2d[4]*i + c_o2d[5]*j + c_o2d[6]*i*j + c_o2d[7];

      int r = floor(x_val);
      int c = floor(y_val);
      if(r<0 || r>=nRows || c<0 || c>=nCols) continue;
      double dr = x_val - r;
      double dc = y_val - c;

      for(int k=0;k<I.channels();k++) p[j][k] = bilinearInterpolation(I, r,c,dr,dc,k);
    }
  }

  return output; 
}

};

class Transformations{
protected:
  int* lookuptable_generator(double *x_coordinate, double *y_coordinate, int n){
    int *lookuptable = new int[256];
    int k = 0;
    double prev_x = 0, prev_y = 0, slope;
    for(int i=0;i<n;i++){
      slope = (y_coordinate[i] - prev_y)/(x_coordinate[i] - prev_x);
      // equation y = m(x-px) + py
      int k_limit = floor(x_coordinate[i]);
      while(k<=k_limit){
        int intensity = round(slope*(k-prev_x) + prev_y);
        lookuptable[k++] = (intensity > 255 ? 255 : intensity );
      }
      prev_x = x_coordinate[i];
      prev_y = y_coordinate[i];
    }
    slope = (255 - prev_y)/(255 - prev_x);
    while(k<256){
      int intensity = round(slope*(k-prev_x) + prev_y);
      lookuptable[k++] = (intensity > 255 ? 255 : intensity );
    } 

    return lookuptable;
  }
};

class IntensityTransformations: Transformations{

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

  Mat logTransformation(Mat &I, double c = 1.0){

    int nRows = I.rows;
    int nCols = I.cols;
    int channels = I.channels();

    int lookuptable_log[256];
    for(int i=0;i<256;i++) lookuptable_log[i] = round(c*log(1+i));


    Mat I_LAB;
    cvtColor(I, I_LAB, COLOR_BGR2Lab);

    Vec3b *p;

    for(int i=0;i<nRows;i++){
      p = I_LAB.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        int intensity = lookuptable_log[p[j][0]];
        p[j][0] = (intensity > 255 ? 255 : intensity );
      }
    }
    cvtColor(I_LAB, I_LAB, CV_Lab2BGR);
    return I_LAB;
  }

Mat gammaCorrection(Mat &I, double gamma = 1.0, double c = 1.0){

    int nRows = I.rows;
    int nCols = I.cols;

    int lookuptable_gamma[256];
    for(int i=0;i<256;i++) lookuptable_gamma[i] = round(c*pow(i*1.0/255,1/gamma)*255);

    Mat I_LAB;
    cvtColor(I, I_LAB, COLOR_BGR2Lab);

    Vec3b *p;

    for(int i=0;i<nRows;i++){
      p = I_LAB.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        int intensity = lookuptable_gamma[p[j][0]];
        p[j][0] = (intensity > 255 ? 255 : intensity );
      }
    }
    cvtColor(I_LAB, I_LAB, CV_Lab2BGR);
    return I_LAB;
  }

Mat bitplaneSlicing(Mat &I, int bit){
  int bit_sliced = 1 << (bit-1);
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
      q[j][k] = p[j][k]&bit_sliced;
      }
    }
  }
  return output;

}

Mat piecewiseLinearTransformation(Mat &I, double *x_coordinate, double *y_coordinate, int n){
  int *lookuptable = lookuptable_generator(x_coordinate,y_coordinate,n);

  int channels = I.channels();
  int nRows = I.rows;
  int nCols = I.cols;

  Mat I_LAB;
  cvtColor(I, I_LAB, COLOR_BGR2Lab);


  Mat output(nRows,nCols,CV_8UC3);
  Vec3b *p,*q;

  for(int i=0;i<nRows;i++){
    p = I_LAB.ptr<Vec3b>(i);
    for(int j=0;j<nCols;j++){
      p[j][0] = lookuptable[p[j][0]];      
    }
  }
  delete[] lookuptable;
  cvtColor(I_LAB, I_LAB, CV_Lab2BGR);
  return I_LAB;
}

Mat histogramEqualization(Mat &I){
    int nRows = I.rows;
    int nCols = I.cols;

    Mat output(nRows,nCols,CV_8UC3);

    Vec3b *p,*q;

    int frequency[256][3] = {};
   

    int lookuptable[256][3];

    for(int i=0;i<nRows;i++){
      p = I.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        for(int k=0;k<I.channels();k++) frequency[p[j][k]][k]++;
      }
    }

    int cumulative_sum[3] = {0};
    for(int i=0;i<256;i++){
      for(int k=0;k<I.channels();k++){
        cumulative_sum[k] += frequency[i][k];
        lookuptable[i][k] = round(255*(cumulative_sum[k])*1.0/(nRows*nCols));
      } 

    }


    for(int i=0;i<nRows;i++){
      p = I.ptr<Vec3b>(i);
      q = output.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        for(int k=0;k<I.channels();k++) q[j][k] = lookuptable[p[j][k]][k];
      }
    }
    return output;
  }

  Mat histogramEqualizationLuminance(Mat &I){
    int nRows = I.rows;
    int nCols = I.cols;

    Mat I_LAB;
    cvtColor(I, I_LAB, COLOR_BGR2Lab);

    Vec3b *p,*q;

    int frequency[256] = {0};
   
    int lookuptable[256];

    for(int i=0;i<nRows;i++){
      p = I.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        frequency[p[j][0]]++;
      }
    }

    int cumulative_sum = 0;
    for(int i=0;i<256;i++){
        cumulative_sum += frequency[i];
        lookuptable[i] = round(255*(cumulative_sum)*1.0/(nRows*nCols));
    }

    for(int i=0;i<nRows;i++){
      p = I_LAB.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        p[j][0] = lookuptable[p[j][0]];
      }
    }
    cvtColor(I_LAB, I_LAB, CV_Lab2BGR);
    return I_LAB;
  }

   Mat histogramMatching(Mat &I, Mat &D){

    int nRows = I.rows;
    int nCols = I.cols;

    int dRows = D.rows;
    int dCols = D.cols;

    Mat output(nRows,nCols,CV_8UC3);

    int lookuptable[256][3] = {};

    double cdf_i[256][3] = {};
    int frequency_i[256][3] = {};
    double cdf_d[256][3] = {};
    int frequency_d[256][3] = {};

    Vec3b *p, *q;

    for(int i=0;i<nRows;i++){
      p = I.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        for(int k=0;k<I.channels();k++) frequency_i[p[j][k]][k]++;
      }
    }

    for(int i=0;i<dRows;i++){
      p = D.ptr<Vec3b>(i);
      for(int j=0;j<dCols;j++){
        for(int k=0;k<D.channels();k++) frequency_d[p[j][k]][k]++;
      }
    }

    int sum_i[3] = {}, sum_d[3] = {};
    for(int i=0;i<256;i++){
      for(int k=0;k<I.channels();k++){
        sum_i[k] += frequency_i[i][k];
        sum_d[k] += frequency_d[i][k];
        cdf_i[i][k] = sum_i[k]*1.0/(nRows*nCols);
        cdf_d[i][k] = sum_d[k]*1.0/(dRows*dCols);
      }
    }

    int newVal[3];
    for(int i=0;i<256;i++){
      double min_diff[3] = {2,2,2}, diff[3];
      for(int j=0;j<256;j++){
        for(int k=0;k<I.channels();k++){

          diff[k] = abs(cdf_i[i][k] - cdf_d[j][k]);
          if(diff[k] < min_diff[k]){
            min_diff[k] = diff[k];
            newVal[k] = j;
          }
        }
      }
      for(int k=0;k<I.channels();k++) lookuptable[i][k] = newVal[k];
    }


    for(int i=0;i<nRows;i++){
      p = I.ptr<Vec3b>(i);
      q = output.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        for(int k=0;k<I.channels();k++) q[j][k] = lookuptable[p[j][k]][k];
      }
    }
    return output;
  }

  Mat histogramMatchingLuminance(Mat &I, Mat &D){

    int nRows = I.rows;
    int nCols = I.cols;

    int dRows = D.rows;
    int dCols = D.cols;

    Mat I_LAB, D_LAB;
    cvtColor(I, I_LAB, CV_BGR2Lab );
    cvtColor(D, D_LAB, CV_BGR2Lab );

    int lookuptable[256] = {0};

    double cdf_i[256] = {0};
    int frequency_i[256] = {0};
    double cdf_d[256] = {0};
    int frequency_d[256] = {0};

    Vec3b *p, *q;

    for(int i=0;i<nRows;i++){
      p = I_LAB.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        frequency_i[p[j][0]]++;
      }
    }

    for(int i=0;i<dRows;i++){
      p = D_LAB.ptr<Vec3b>(i);
      for(int j=0;j<dCols;j++){
        frequency_d[p[j][0]]++;
      }
    }

    int sum_i = 0, sum_d = 0;
    for(int i=0;i<256;i++){
      sum_i += frequency_i[i];
      sum_d += frequency_d[i];
      cdf_i[i] = sum_i*1.0/(nRows*nCols);
      cdf_d[i] = sum_d*1.0/(dRows*dCols);
    }

    double min_diff, diff;
    int newVal;
    for(int i=0;i<256;i++){
      min_diff = 2; // inf
      for(int j=0;j<256;j++){
        diff = abs(cdf_i[i] - cdf_d[j]);
        if(diff < min_diff){
          min_diff = diff;
          newVal = j;
        }
      }
      lookuptable[i] = newVal;
    }


    for(int i=0;i<nRows;i++){
      p = I_LAB.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        p[j][0] = lookuptable[p[j][0]];
      }
    }
    cvtColor(I_LAB, I_LAB, CV_Lab2BGR);
    return I_LAB;
  }

  Mat adaptiveHistogramEquilization(Mat &I, int tile, int grid_x, int grid_y){
    // Do local he for tile/2, tile/2, then interpolate.

    int nRows = I.rows;
    int nCols = I.cols;

    Mat I_gray(I);

    Mat I_LAB, D_LAB;
    cvtColor(I, I_LAB, CV_BGR2Lab );

    int size[3] = {grid_x, grid_y, 256};
    Mat grid(3, size, CV_8UC(1));
  

    Vec3b *p,*q, *t, *g;

    double scale_x = nRows*1.0/(grid_x);
    double scale_y = nCols*1.0/(grid_y);
    for(int i=0;i<grid_x;i++){

      for(int j=0;j<grid_y;j++){
        // For each grid
        int frequency[256] = {0};
        int lookuptable[256];

        int origin_i = round(i*scale_x);
        int origin_j = round(j*scale_y);

        int start_x = origin_i - tile/2, start_y = origin_j - tile/2;
        int end_x = origin_i + tile/2, end_y = origin_j + tile/2;

        int dim_x = 0;
        int dim_y = 0;
        for(int x = (start_x<0? 0 : start_x); x < (end_x >= nRows? nRows: end_x); x++){
          p = I_LAB.ptr<Vec3b>(x);
          dim_y = 0;
          for(int y = (start_y<0? 0 : start_y ); y < (end_y >= nCols? nCols: end_y); y++){
            frequency[p[y][0]]++;
            dim_y++;
          }
          dim_x++;
        }

        int cumulative_sum = 0;

        for(int x=0;x<256;x++){
          cumulative_sum += frequency[x];
          lookuptable[x] = round(255*(cumulative_sum)*1.0/(dim_x*dim_y));
          grid.at<uchar>(i,j,x) = lookuptable[x];
        }
      }
    }

    // Iterpolation
    for(int i=0;i<nRows;i++){
      p = I_LAB.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){

        int r = floor(i/scale_x);
        int c = floor(j/scale_y);

        r = (r<0?0:r);
        r = (r>grid_x-2?grid_x-2:r);

        c = (c<0?0:c);
        c = (c>grid_y-2?grid_y-2:c);
        
        int dr = i/scale_x - r;
        int dc = j/scale_y - c;
        int in = p[j][0];
        double val = grid.at<uchar>(r,c,in)*(1-dr)*(1-dc);
        val += grid.at<uchar>(r+1,c,in)*(dr)*(1-dc);
        val += grid.at<uchar>(r,c+1,in)*(dc)*(1-dr);
        val += grid.at<uchar>(r+1,c+1,in)*(dr)*(dc);
        p[j][0] = round(val);

      }
    }
    cvtColor(I_LAB, I_LAB, CV_Lab2BGR);
    return I_LAB;
  }

  Mat adaptiveHistogramEquilizationWindow(Mat &I, int tile){

    int nRows = I.rows;
    int nCols = I.cols;

    Mat I_gray(I);
    Mat I_LAB, D_LAB;
    cvtColor(I, I_LAB, CV_BGR2Lab );

    Mat T(I_LAB);

    Vec3b *p,*q, *t, *g;

    int intensity = 0;
    for(int i=0;i<nRows;i++){
      q = T.ptr<Vec3b>(i);
      for(int j=0;j<nCols;j++){
        // For each grid
        int frequency[256] = {0};
        int lookuptable[256];

        int start_x = i - tile/2, start_y = j - tile/2;
        int end_x = i + tile/2, end_y = j + tile/2;

        int dim_x = 0;
        int dim_y = 0;
        for(int x = (start_x<0? 0 : start_x); x < (end_x >= nRows? nRows: end_x); x++){
          p = I_LAB.ptr<Vec3b>(x);
          dim_y = 0;
          for(int y = (start_y<0? 0 : start_y ); y < (end_y >= nCols? nCols: end_y); y++){
            frequency[p[y][0]]++;
            dim_y++;
          }
          dim_x++;
        }

        int cumulative_sum = 0;

        for(int x=0;x<256;x++){
          cumulative_sum += frequency[x];
          lookuptable[x] = round(255*(cumulative_sum)*1.0/(dim_x*dim_y));
        }
        q[j][0] = lookuptable[q[j][0]];
      }
    }
    cvtColor(T,T, CV_Lab2BGR);
    return T;
  }

};

class IntensityTransformationsGray: Transformations{

int bilinearInterpolation(Mat &I, int r, int c, double dr, double dc){

    double val = I.at<uchar>(r,c)*(1-dr)*(1-dc);
    val += I.at<uchar>(r+1,c)*(dr)*(1-dc);
    val += I.at<uchar>(r,c+1)*(1-dr)*(dc);
    val += I.at<uchar>(r+1,c+1)*(dr)*dc;

    int intensity = round(val);

    return intensity;
  }

public:
  Mat gammaCorrection(Mat &I,  double gamma = 1.0, double c = 1.0){

    int nRows = I.rows;
    int nCols = I.cols;

    int lookuptable_gamma[256];
    for(int i=0;i<256;i++) lookuptable_gamma[i] = round(c*pow(i*1.0/255,1/gamma)*255);

    Mat I_gray(I);
    if(I.channels() == 3) cvtColor(I, I_gray, CV_BGR2GRAY );

    Mat output(nRows,nCols,CV_8UC1);

    uchar *p,*q;

    for(int i=0;i<nRows;i++){
      p = I_gray.ptr<uchar>(i);
      q = output.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){
        int intensity = lookuptable_gamma[p[j]];
        q[j] = (intensity > 255 ? 255 : intensity );
      }
    }
    return output;
  }

Mat piecewiseLinearTransformation(Mat &I, double *x_coordinate, double *y_coordinate, int n){
  int *lookuptable = lookuptable_generator(x_coordinate,y_coordinate,n);

  int channels = I.channels();
  int nRows = I.rows;
  int nCols = I.cols;

  Mat I_gray(I);

   if(I.channels() == 3) cvtColor(I, I_gray, CV_BGR2GRAY );

  Mat output(nRows,nCols,CV_8UC1);

  uchar *p,*q;

  for(int i=0;i<nRows;i++){
    p = I_gray.ptr<uchar>(i);
    q = output.ptr<uchar>(i);
    for(int j=0;j<nCols;j++){
      q[j] = lookuptable[p[j]];
    }
  }

  delete[] lookuptable;
  return output;
}

Mat logTransformation(Mat &I, double c = 1.0){

    int nRows = I.rows;
    int nCols = I.cols;  


    int lookuptable_log[256];
    for(int i=0;i<256;i++) lookuptable_log[i] = round(c*log(1+i));

    Mat I_gray(I);

     if(I.channels() == 3) cvtColor(I, I_gray, CV_BGR2GRAY );

    Mat output(nRows,nCols,CV_8UC1);

    uchar *p,*q;

    for(int i=0;i<nRows;i++){
      p = I_gray.ptr<uchar>(i);
      q = output.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){
        int intensity = lookuptable_log[p[j]];
        q[j] = (intensity > 255 ? 255 : intensity );
      }
    }

    return output;
  }

  Mat negative(Mat &I){
    // Returns the negative of the image 
    int nRows = I.rows;
    int nCols = I.cols;

    Mat I_gray(I);

     if(I.channels() == 3) cvtColor(I, I_gray, CV_BGR2GRAY );

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

Mat histogramEqualization(Mat &I){
    int nRows = I.rows;
    int nCols = I.cols;

    Mat I_gray(I);

    if(I.channels() == 3) cvtColor(I, I_gray, CV_BGR2GRAY );

    Mat output(nRows,nCols,CV_8UC1);

    uchar *p,*q;

    int frequency[256] = {0};
    int lookuptable[256];

    for(int i=0;i<nRows;i++){
      p = I_gray.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){
        frequency[p[j]]++;
      }
    }

    int cumulative_sum = 0;

    for(int i=0;i<256;i++){
      cumulative_sum += frequency[i];
      lookuptable[i] = round(255*(cumulative_sum)*1.0/(nRows*nCols));
    }


    for(int i=0;i<nRows;i++){
      p = I_gray.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){
        p[j] = lookuptable[p[j]];
      }
    }
    return I_gray;
  }

  Mat histogramMatching(Mat &I, Mat &D){

    int nRows = I.rows;
    int nCols = I.cols;
    int dRows = D.rows;
    int dCols = D.cols;

    Mat I_gray(I), D_gray(D);

    if(I.channels() == 3) cvtColor(I, I_gray, CV_BGR2GRAY );
    if(D.channels() == 3) cvtColor(D, D_gray, CV_BGR2GRAY );

    Mat output(nRows,nCols,CV_8UC1);

    int lookuptable[256] = {0};

    double cdf_i[256] = {0};
    int frequency_i[256] = {0};
    double cdf_d[256] = {0};
    int frequency_d[256] = {0};

    uchar *p, *q;

    for(int i=0;i<nRows;i++){
      p = I_gray.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){
        frequency_i[p[j]]++;
      }
    }

    for(int i=0;i<dRows;i++){
      p = D_gray.ptr<uchar>(i);
      for(int j=0;j<dCols;j++){
        frequency_d[p[j]]++;
      }
    }

    int sum_i = 0, sum_d = 0;
    for(int i=0;i<256;i++){
      sum_i += frequency_i[i];
      sum_d += frequency_d[i];
      cdf_i[i] = sum_i*1.0/(nRows*nCols);
      cdf_d[i] = sum_d*1.0/(dRows*dCols);
    }

    double min_diff, diff;
    int newVal;
    for(int i=0;i<256;i++){
      min_diff = 2; // inf
      for(int j=0;j<256;j++){
        diff = abs(cdf_i[i] - cdf_d[j]);
        if(diff < min_diff){
          min_diff = diff;
          newVal = j;
        }
      }
      lookuptable[i] = newVal;
    }

    for(int i=0;i<nRows;i++){
      p = I_gray.ptr<uchar>(i);
      q = output.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){
        q[j] = lookuptable[p[j]];
      }
    }
    return output;
  }

  Mat adaptiveHistogramEquilizationWindow(Mat &I, int tile){

    int nRows = I.rows;
    int nCols = I.cols;

    Mat I_gray(I);

    if(I.channels() == 3) cvtColor(I, I_gray, CV_BGR2GRAY );

    Mat T(I_gray);

    uchar *p,*q, *t, *g;

    int intensity = 0;
    for(int i=0;i<nRows;i++){
      q = T.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){
        // For each grid
        int frequency[256] = {0};
        int lookuptable[256];

        int start_x = i - tile/2, start_y = j - tile/2;
        int end_x = i + tile/2, end_y = j + tile/2;

        int dim_x = 0;
        int dim_y = 0;
        for(int x = (start_x<0? 0 : start_x); x < (end_x >= nRows? nRows: end_x); x++){
          p = I_gray.ptr<uchar>(x);
          dim_y = 0;
          for(int y = (start_y<0? 0 : start_y ); y < (end_y >= nCols? nCols: end_y); y++){
            frequency[p[y]]++;
            dim_y++;
          }
          dim_x++;
        }

        int cumulative_sum = 0;

        for(int x=0;x<256;x++){
          cumulative_sum += frequency[x];
          lookuptable[x] = round(255*(cumulative_sum)*1.0/(dim_x*dim_y));
        }
        q[j] = lookuptable[q[j]];
      }
    }
    return T;
  }


  Mat adaptiveHistogramEquilization(Mat &I, int tile, int grid_x, int grid_y){
    // Do local he for tile/2, tile/2, then interpolate.

    int nRows = I.rows;
    int nCols = I.cols;

    Mat I_gray(I);

    if(I.channels() == 3) cvtColor(I, I_gray, CV_BGR2GRAY );
    int size[3] = {grid_x, grid_y, 256};
    Mat grid(3, size, CV_8UC(1));
  

    uchar *p,*q, *t, *g;

    double scale_x = nRows*1.0/(grid_x);
    double scale_y = nCols*1.0/(grid_y);
    for(int i=0;i<grid_x;i++){

      for(int j=0;j<grid_y;j++){
        // For each grid
        int frequency[256] = {0};
        int lookuptable[256];

        int origin_i = round(i*scale_x);
        int origin_j = round(j*scale_y);

        int start_x = origin_i - tile/2, start_y = origin_j - tile/2;
        int end_x = origin_i + tile/2, end_y = origin_j + tile/2;

        int dim_x = 0;
        int dim_y = 0;
        for(int x = (start_x<0? 0 : start_x); x < (end_x >= nRows? nRows: end_x); x++){
          p = I_gray.ptr<uchar>(x);
          dim_y = 0;
          for(int y = (start_y<0? 0 : start_y ); y < (end_y >= nCols? nCols: end_y); y++){
            frequency[p[y]]++;
            dim_y++;
          }
          dim_x++;
        }

        int cumulative_sum = 0;

        for(int x=0;x<256;x++){
          cumulative_sum += frequency[x];
          lookuptable[x] = round(255*(cumulative_sum)*1.0/(dim_x*dim_y));
          grid.at<uchar>(i,j,x) = lookuptable[x];
        }
      }
    }

    // Iterpolation
    for(int i=0;i<nRows;i++){
      p = I_gray.ptr<uchar>(i);
      for(int j=0;j<nCols;j++){

        int r = floor(i/scale_x);
        int c = floor(j/scale_y);

        r = (r<0?0:r);
        r = (r>grid_x-2?grid_x-2:r);

        c = (c<0?0:c);
        c = (c>grid_y-2?grid_y-2:c);
        
        int dr = i/scale_x - r;
        int dc = j/scale_y - c;
        int in = p[j];
        double val = grid.at<uchar>(r,c,in)*(1-dr)*(1-dc);
        val += grid.at<uchar>(r+1,c,in)*(dr)*(1-dc);
        val += grid.at<uchar>(r,c+1,in)*(dc)*(1-dr);
        val += grid.at<uchar>(r+1,c+1,in)*(dr)*(dc);
        p[j] = round(val);

      }
    }
    return I_gray;
  }


Mat bitplaneSlicing(Mat &I, int bit){

  int bit_sliced = 1 << (bit-1);
  int nRows = I.rows;
  int nCols = I.cols;

  Mat I_gray(I);

  if(I.channels() == 3) cvtColor(I, I_gray, CV_BGR2GRAY );
  Mat output(nRows,nCols,CV_8UC1);

  uchar *p,*q;

  for(int i=0;i<nRows;i++){
    p = I_gray.ptr<uchar>(i);
    q = output.ptr<uchar>(i);
    for(int j=0;j<nCols;j++){
      
      q[j] = p[j]&bit_sliced;
      
    }
  }
  return output;

}


};



class Menu{

  bool check(Mat I){
    if(! I.data ) {
       cout <<  "Could not open or find the image" << endl ;
       return false;
     }
     return true;
  }

  void resize(){
    Mat image;
    Mat output;
    cout << "imresize>> File Path | Factor | Interpolation (nearest or bilinear)"<<endl;
    string file, interpolation;
    double factor;
    cin>>file>>factor>>interpolation;

    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    AffineTransformation at;

    if(interpolation == "nearest") output = at.resize(image, factor, nearest);
    else if(interpolation == "bilinear") output = at.resize(image, factor, bilinear);
    else{
      cout << "Invalid Command";
      return;
    } 
    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void scale(){
    Mat image;
    Mat output;
    cout << "imscale>> File Path | Factor X | Factor Y | Interpolation (nearest or bilinear)"<<endl;
    string file, interpolation;
    double factor_x, factor_y;
    cin>>file>>factor_x>>factor_y>>interpolation;

    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    AffineTransformation at;

    if(interpolation == "nearest") output = at.scale(image, factor_x, factor_y, nearest);
    else if(interpolation == "bilinear") output = at.scale(image, factor_x, factor_y, bilinear);
    else{
      cout << "Invalid Command";
      return;
    } 
    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void rotate(){
    Mat image;
    Mat output;
    cout << "imrotate>> File Path | Degree | Interpolation (nearest or bilinear)"<<endl;
    string file, interpolation;
    double factor;
    cin>>file>>factor>>interpolation;

    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    AffineTransformation at;

    if(interpolation == "nearest") output = at.rotate(image, factor, nearest);
    else if(interpolation == "bilinear") output = at.rotate(image, factor, bilinear);
    else{
      cout << "Invalid Command";
      return;
    } 
    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void translate(){
    Mat image;
    Mat output;
    cout << "imtranslate>> File Path | Factor X | Factor Y "<<endl;
    string file;
    int factor_x, factor_y;
    cin>>file>>factor_x>>factor_y;

    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    AffineTransformation at;

    output = at.translate(image, factor_x, factor_y);
   
    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

   void shear(){
    Mat image;
    Mat output;
    cout << "imshear>> File Path | Axis (x or y) | Factor | Interpolation"<<endl;
    string file, interpolation;
    double factor;
    char axis;
    cin>>file>>axis>>factor>>interpolation;

    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    AffineTransformation at;

    if(interpolation == "nearest") output = at.shear(image, factor, axis, nearest);
    else if(interpolation == "bilinear") output = at.shear(image, factor, axis, bilinear);
    else{
      cout << "Invalid Command";
      return;
    } 
   
    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void tie(){
    Mat image;
    Mat output;
    cout << "imrecon>> File Path | Original Image [(x1,y1), .. (x4,y4)] | Distorted Image [(x1,y1), .. (x4,y4)]"<<endl;
    string file;
    int x_orig[4];
    int y_orig[4];
    int x_dist[4];
    int y_dist[4];
    cin>>file;

    for(int i=0;i<4;i++) cin>>x_orig[i]>>y_orig[i];
    for(int i=0;i<4;i++) cin>>x_dist[i]>>y_dist[i];

    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    AffineTransformation at;

    output = at.tiePoints(image, x_orig, y_orig, x_dist, y_dist);
   
    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void negative(bool rgb){
    Mat image;
    Mat output;
    cout << "negative>> File Path "<<endl;
    string file;
    cin>>file;

    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    if(rgb){
      IntensityTransformations it;
      output = it.negative(image);
    } 
    else{
      IntensityTransformationsGray it;
      output = it.negative(image);
    }
   
    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void log_t(bool rgb){
    Mat image;
    Mat output;
    cout << "log>> File Path | c "<<endl;
    string file;
    double c;
    cin>>file>>c;

    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    if(rgb){
      IntensityTransformations it;
      output = it.logTransformation(image, c);
    } 
    else{
      IntensityTransformationsGray it;
      output = it.logTransformation(image, c);
    }

   
    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void gamma_t(bool rgb){
    Mat image;
    Mat output;
    cout << "gamma>> File Path | gamma | c "<<endl;
    string file;
    double c, gamma;
    cin>>file>>gamma>>c;

    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    if(rgb){
      IntensityTransformations it;
      output = it.gammaCorrection(image, gamma, c);
    } 
    else{
      IntensityTransformationsGray it;
      output = it.gammaCorrection(image, gamma, c);
    }

   
    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void linear_t(bool rgb){
    Mat image;
    Mat output;
    cout << "linear>> File Path | number of points | points (xi,yi) "<<endl;
    string file;
    int n;
    cin>>file>>n;
    double x[n];
    double y[n];
    for(int i=0;i<n;i++){
      cin>>x[i]>>y[i];
    }

    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    if(rgb){
      IntensityTransformations it;
      output = it.piecewiseLinearTransformation(image, x, y, n);
    } 
    else{
      IntensityTransformationsGray it;
      output = it.piecewiseLinearTransformation(image, x, y, n);
    }

    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void bit_t(bool rgb){
    Mat image;
    Mat output;
    cout << "bitslice>> File Path | Bit(1-8) "<<endl;
    string file;
    int n;
    cin>>file>>n;
   
    image = imread(file.c_str() , 1); 

    if(!check(image)) return;

    if(rgb){
      IntensityTransformations it;
      output = it.bitplaneSlicing(image, n);
    } 
    else{
      IntensityTransformationsGray it;
      output = it.bitplaneSlicing(image, n);
    }

    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void hist(bool rgb){
    Mat image;
    Mat output;

    if(rgb){
      cout << "hist>> File Path | Channels (1:Luminance or 3:RGB) "<<endl;
      string file;
      int n;
      cin>>file>>n;
     
      image = imread(file.c_str() , 1); 

      if(!check(image)) return;

      IntensityTransformations it;
      if(n == 1) output = it.histogramEqualizationLuminance(image);
      else if(n == 3) output = it.histogramEqualization(image);
    } 
    else{
      cout << " File Path "<<endl;
      string file;
      cin>>file;
     
      image = imread(file.c_str() , 1); 
      IntensityTransformationsGray it;
      output = it.histogramEqualization(image);
    }

    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void adapthist(bool rgb){
    Mat image;
    Mat output;

    if(rgb){
      cout << "adapthist>> File Path | Method (window or tile) "<<endl;
      string file, method;
      cin>>file>>method;
      
      image = imread(file.c_str() , 1); 

      if(!check(image)) return;

      IntensityTransformations it;

      if(method == "window"){
          cout << " window -> Window Size (n: where window is nxn) "<<endl;
          int n;
          cin>>n;
          output = it.adaptiveHistogramEquilizationWindow(image, n);
      }

      else if(method == "tile"){
          cout << " tile -> Tile Size (n: where window is nxn) | Grid X | Grid Y "<<endl;
          int n,x,y;
          cin>>n>>x>>y;
          output = it.adaptiveHistogramEquilization(image, n, x, y);
      }
    } 
    else{
      cout << "adapthist>> File Path | Method (window or tile) "<<endl;
      string file, method;
      cin>>file>>method;
      
      image = imread(file.c_str() , 1); 
      IntensityTransformationsGray it;

      if(method == "window"){
          cout << " window -> Window Size (n: where window is nxn) "<<endl;
          int n;
          cin>>n;
          output = it.adaptiveHistogramEquilizationWindow(image, n);
      }

      else if(method == "tile"){
          cout << " tile -> Tile Size (n: where window is nxn) | Grid X | Grid Y "<<endl;
          int n,x,y;
          cin>>n>>x>>y;
          output = it.adaptiveHistogramEquilization(image, n, x, y);
      }
    }

    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }

  void histmatch(bool rgb){
    Mat image_input, image_reference;
    Mat output;

    if(rgb){
      cout << "histmatch>> File Path (input) | File Path (reference) | Channels (1:Luminance or 3:RGB) "<<endl;
      string file_input, file_reference;
      int n;
      cin>>file_input>>file_reference>>n;
     
      image_input = imread(file_input.c_str() , 1); 
      if(!check(image_input)) return;
      image_reference = imread(file_reference.c_str() , 1);
      if(!check(image_reference)) return;

      IntensityTransformations it;
      if(n == 1) output = it.histogramMatchingLuminance(image_input, image_reference);
      else if(n == 3) output = it.histogramMatching(image_input, image_reference);
    } 
    else{
      cout << "histmatch>> File Path (input) | File Path (reference) "<<endl;
      string file_input, file_reference;
      cin>>file_input>>file_reference;
     
      image_input = imread(file_input.c_str() , 1); 
      image_reference = imread(file_reference.c_str() , 1);

      IntensityTransformationsGray it;
      output = it.histogramMatching(image_input, image_reference);
    }

    imshow( "Display window", output);  
    waitKey(0);
    destroyWindow("Display window");
  }


public:
  void help(){
    cout << endl;
    cout << " Image Resize (Uniform Scale) : imresize"<<endl;
    cout << " Image Scale (Non-Uniform Scale) : imscale" << endl;
    cout << " Image Rotation (Degrees) : imrotate "<<endl;
    cout << " Image Translation (By pixel) : imtranslate"<<endl;
    cout << " Image Shear : imshear"<<endl;
    cout << " Image Reconstruction (Tie Points) : imrecon"<<endl;
    cout << " Negative : negative"<<endl;
    cout << " Log Transformation : log"<<endl;
    cout << " Gamma Transformation : gamma"<<endl;
    cout << " Piecewise Linear Transformation : linear"<<endl;
    cout << " Bitplane Slicing : bitslice"<<endl;
    cout << " Histogram Equalization : histequal "<<endl;
    cout << " Adaptive Histogram Equalization : adapthist"<<endl;
    cout << " Histogram Matching : histmatch"<<endl<<endl;
    cout << " To work on RGB, type rgb. For Grayscale type gray. By default, its grayscale."<<endl<<endl;
    cout << " To see the list again, just type help"<<endl;
    cout << " To exit, type exit"<<endl;
  
  }


  void init(){
    bool rgb = false;
    bool run = true;
    string s, mode = "Gray";

    help();
    while(run){
      if(rgb) mode = "RGB";
      else mode = "Gray";
      cout<<endl<<mode<<">> ";
      cin>>s;
      if(s == "help") help();
      else if(s == "imresize") resize();
      else if(s == "imscale") scale();
      else if(s == "imrotate") rotate();
      else if(s == "imtranslate") translate();
      else if(s == "imshear") translate();
      else if(s == "imrecon") tie();
      else if(s == "negative") negative(rgb);
      else if(s == "log") log_t(rgb);
      else if(s == "gamma") gamma_t(rgb);
      else if(s == "linear") linear_t(rgb);
      else if(s == "bitslice") bit_t(rgb);
      else if(s == "histequal") hist(rgb);
      else if(s == "adapthist") adapthist(rgb);
      else if(s == "histmatch") histmatch(rgb);
      else if(s == "rgb") rgb = true;
      else if(s == "gray") rgb = false;
      else if(s == "exit") run = false;
      else cout<<">> Invalid Command. Type help for help."<<endl;
    }
  }
};

int main( int argc, char** argv ) {
  

  
  // if(! image.data ) {
  //     cout <<  "Could not open or find the image" << endl ;
  //     return -1;
  //   }
  Menu m;
  m.init();
  
  waitKey(0);
  return 0;
}
