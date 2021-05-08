// ISE(Lab2).cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cmath>
#include "highgui/highgui.hpp"
#include "core.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
// code an RGB to GREY

Mat RGBtoGrey(Mat RGB) {
    Mat Grey = Mat::zeros(RGB.size(), CV_8UC1);
    // 8 bits one channel, so compiler knows, pixel only one value
    // changng CV_8UC3 then it turns tousing 3 columns like RGB
    for (int i = 0; i < RGB.rows; i++) {
        for (int j = 0; j < RGB.cols * 3; j = j + 3) {
            Grey.at<uchar>(i, j / 3) = ( RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2))/ 3;
        }
    }
    return Grey;
}

//code an grey to Binary
Mat GreytoBinary(Mat Grey, int Threshold) {
    Mat Binary = Mat::zeros(Grey.size(), CV_8UC1);
    //if the grey is more than 127 then white "255" else black
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            if (Grey.at<uchar>(i, j) > Threshold) {
                Binary.at<uchar>(i, j) = 255;
            }
        }
    }
    return Binary;
}

// Math functions
//Invert grey img 
Mat Invert(Mat Grey) {
    Mat Inverted = Mat::zeros(Grey.size(), CV_8UC1);
    //255 - input pixel
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            Inverted.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);
        }
    }
    return Inverted;
}

//Darken the grey img 
Mat Darken(Mat Grey, int Threshold) {
    Mat DarkenImg = Mat::zeros(Grey.size(), CV_8UC1);
    //if value more than threshold, it will become 100
    // else its the same
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            if (Grey.at<uchar>(i, j) >= Threshold) {
                DarkenImg.at<uchar>(i, j) = Threshold;
            }
            else {
                DarkenImg.at<uchar>(i, j) = Grey.at<uchar>(i, j);
            }
        }
    }
    return DarkenImg;
}

//Step function img 
Mat StepFunction(Mat Grey, int th1, int th2) {
    Mat outputImg = Mat::zeros(Grey.size(), CV_8UC1);

    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            if (Grey.at<uchar>(i, j) >= th1 && Grey.at<uchar>(i, j) <= th2) {
                outputImg.at<uchar>(i, j) = 255;
            }
            else {
                outputImg.at<uchar>(i, j) = 0;
            }
        }
    }
    return outputImg;
}

//Max function 
Mat MaxFunction(Mat Grey, int iNeighbour) {
    // Clone it so border remains
    // can assign zeros as well since border wont affect much
    Mat outputImg = Grey.clone();

    // Start from second row / column
    // Stop before the last row / column
    for (int i = iNeighbour; i < Grey.rows - iNeighbour; i++) {
        for (int j = iNeighbour; j < Grey.cols - iNeighbour; j++) {

            //loop check max
            int max = -1;
            for (int c = -iNeighbour; c <= iNeighbour; c++) {
                for (int r = -iNeighbour; r <= iNeighbour; r++) {
                    if (Grey.at<uchar>(i + r, j + c) > max) {
                        max = Grey.at<uchar>(i + r, j + c);
                        //std::cout << max;
                    }
                }
            }     
            outputImg.at<uchar>(i, j) = max;
        }
    }
    return outputImg;
}

// Min function
Mat MinFunction(Mat Grey, int iNeighbour) {
    Mat outputImg = Mat::zeros(Grey.size(), CV_8UC1);;

    // Start from second row / column
    // Stop before the last row / column
    for (int i = iNeighbour; i < Grey.rows - iNeighbour; i++) {
        for (int j = iNeighbour; j < Grey.cols - iNeighbour; j++) {

            //loop check max
            int min = 256;
            for (int c = -iNeighbour; c <= iNeighbour; c++) {
                for (int r = -iNeighbour; r <= iNeighbour; r++) {
                    if (Grey.at<uchar>(i + r, j + c) < min) {
                        min = Grey.at<uchar>(i + r, j + c);

                    }
                }
            }
            outputImg.at<uchar>(i, j) = min;
        }
    }
    return outputImg;
}

// Avg function
Mat AverageFunction(Mat Grey, int iNeighbour) {
    Mat outputImg = Mat::zeros(Grey.size(), CV_8UC1);;

    // Start from second row / column
    // Stop before the last row / column
    for (int i = iNeighbour; i < Grey.rows - iNeighbour; i++) {
        for (int j = iNeighbour; j < Grey.cols - iNeighbour; j++) {

            //loop sum all neighbour
            int sum = 0;
            int counter = 0;
            for (int c = -iNeighbour; c <= iNeighbour; c++) {
                for (int r = -iNeighbour; r <= iNeighbour; r++) {
                
                    sum += Grey.at<uchar>(i + r, j + c); 
                    counter++;
                }
            }
            outputImg.at<uchar>(i, j) = (sum / counter);
        }
    }
    return outputImg;
}

// Edge function
Mat EdgeFunction(Mat Grey, int th) {
    Mat outputImg = Mat::zeros(Grey.size(), CV_8UC1);;

    // Start from second row / column
    // Stop before the last row / column
    for (int i = 1; i < Grey.rows - 1; i++) {
        for (int j = 1; j < Grey.cols - 1; j++) {
            int sum = 0;

            sum += (Grey.at<uchar>(i, j) * -4);
            sum += Grey.at<uchar>(i - 1, j);
            sum += Grey.at<uchar>(i, j - 1);
            sum += Grey.at<uchar>(i, j + 1);
            sum += Grey.at<uchar>(i + 1, j);

            if (sum < 0) {       
                outputImg.at<uchar>(i, j) = 0;
            }
            else if (sum > th){
                outputImg.at<uchar>(i, j) = 255;
            }
            else {
                outputImg.at<uchar>(i, j) = sum;
            }
            
        }
    }
    return outputImg;
}

// Equalise function
Mat EqualizeHist(Mat Grey) {
    Mat outputImg = Mat::zeros(Grey.size(), CV_8UC1);;

    //count pixels from 0 to 255
    int count[256] = { 0 };
    float prob[256] = { 0.0 };
    float accuProb[256] = { 0.0 };

    // Loop rows and columns of whole img
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            count[Grey.at<uchar>(i, j)] += 1;
        }
    }

    // Find out the probability
    for (int i = 0; i < 256; i++) {
        // Cast the int to float data type
        // if int divide int output int, so the prob float is 0
        prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);
        
        if (i != 0) {
            accuProb[i] = prob[i] + accuProb[i - 1];
        }
        else {
            accuProb[i] = prob[i];
        }
        
    }

    // Calculating new pixels
    int newPixel[256] = { 0 };
    for (int i = 0; i < 256; i++) {

        newPixel[i] = (256 - 1) * accuProb[i];
    }

    // replace all pixels to new pixel
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            outputImg.at<uchar>(i, j) = newPixel[Grey.at<uchar>(i, j)];
        }
    }
    
    return outputImg;
}

// vertical sobel
Mat VerticalSobel(Mat Grey, int th) {
    Mat outputImg = Mat::zeros(Grey.size(), CV_8UC1);;

    // Start from second row / column
    // Stop before the last row / column
    for (int i = 1; i < Grey.rows - 1; i++) {
        for (int j = 1; j < Grey.cols - 1; j++) {
            int gx = 0;        

            gx = (Grey.at<uchar>(i, j - 1) * -2) + (Grey.at<uchar>(i - 1, j - 1) * -1) + (Grey.at<uchar>(i + 1, j - 1) * -1) + (Grey.at<uchar>(i - 1, j + 1) * 1) + (Grey.at<uchar>(i, j + 1) * 2) + (Grey.at<uchar>(i + 1, j + 1) * 1);

            if (gx <= th) {
                outputImg.at<uchar>(i, j) = 0;
            }
            else {
                outputImg.at<uchar>(i, j) = 255;
            }
           
        }
    }
    return outputImg;
}

// Horizontal sobel
Mat HorizontalSobel(Mat Grey, int Threshold) {
    Mat outputImg = Mat::zeros(Grey.size(), CV_8UC1);;

    // Start from second row / column
    // Stop before the last row / column
    for (int i = 1; i < Grey.rows - 1; i++) {
        for (int j = 1; j < Grey.cols - 1; j++) {
            int gx = 0;

            gx = (Grey.at<uchar>(i -1 , j - 1) * -1) + (Grey.at<uchar>(i - 1, j) * -2) + (Grey.at<uchar>(i - 1, j + 1) * -1) + (Grey.at<uchar>(i + 1, j -1) * 1) + (Grey.at<uchar>(i+1, j ) * 2) + (Grey.at<uchar>(i + 1, j + 1) * 1);

            if (gx <= Threshold) {
                outputImg.at<uchar>(i, j) = 0;
            }
            else {
                outputImg.at<uchar>(i, j) = 255;
            }

        }
    }
    return outputImg;
}

// Dilation
// Check the neighbours, if anyone of them is white
// Change black to white
Mat Dilation(Mat Grey, int neigh) {
    Mat outputImg = Mat::zeros(Grey.size(), CV_8UC1);

    // Start from second row / column
    // Stop before the last row / column
    for (int i = neigh; i < Grey.rows - neigh; i++) {
        for (int j = neigh; j < Grey.cols - neigh; j++) {
            
            for (int r = -neigh; r <= neigh; r++) {
                for (int c = -neigh; c <= neigh; c++) {
                    if (Grey.at<uchar>(i + r, j + c) == 255) {
                        outputImg.at<uchar>(i, j) = 255;
                        break;
                        break;
                    }
                }
            }

        }
    }
    return outputImg;
}

Mat Erosion(Mat Grey, int neigh) {
    Mat outputImg = Grey.clone();
    
    // makes all to 255
    //Mat outputImg = Mat(Grey.size(), CV_8UC1, Scalar(255, 255, 255));

    // Start from second row / column
    // Stop before the last row / column
    for (int i = neigh; i < Grey.rows - neigh; i++) {
        for (int j = neigh; j < Grey.cols - neigh; j++) {

            for (int r = -neigh; r <= neigh; r++) {
                for (int c = -neigh; c <= neigh; c++) {
                    if (Grey.at<uchar>(i + r, j + c) == 0) {
                        outputImg.at<uchar>(i, j) = 0;
                        break;                      
                    }
                }
            }

        }
    }
    return outputImg;
}

int OTSU(Mat Grey) {
    //count pixels from 0 to 255
    int count[256] = { 0 };
    float prob[256] = { 0.0 };
    float accuProb[256] = { 0.0 };

    // Loop rows and columns of whole img
    for (int i = 0; i < Grey.rows; i++) {
        for (int j = 0; j < Grey.cols; j++) {
            count[Grey.at<uchar>(i, j)] += 1;
        }
    }

    for (int i = 0; i < 256; i++)
        prob[i] = (double)count[i] / (double)(Grey.rows * Grey.cols);

    // Find out the probability
    accuProb[0] = prob[0];
    for (int i = 1; i < 256; i++) {
        // Cast the int to float data type
        // if int divide int output int, so the prob float is 0
        accuProb[i] = prob[i] + accuProb[i - 1];
    }

    float meu[256] = { 0.0 };
    // calculate meu = accumulative i * prob[i]
    // no need start from 0, since 0 = 0
    // actual big brain can use i-1 also
    for (int i = 1; i < 256; i++) {
        meu[i] = i * prob[i] + meu[i - 1];
    }

    //do sigma
    float sigma[256] = { 0.0 };
    for (int i = 0; i < 256; i++) {
        sigma[i] = pow((meu[255] * accuProb[i] - meu[i]), 2.0) / (accuProb[i] * (1 - accuProb[i]));
    }

    // find the index (i) that has the max sigma among all the 256 values
    float maxsigma = -1;
    int OTSUTh = 0;
    for (int i = 0; i < 256; i++)
    {
        if (sigma[i] > maxsigma)
        {
            maxsigma = sigma[i];
            OTSUTh = i;
        }
    }

    return OTSUTh;
}

int main()
{
    // Loop every image in dataset for license plate detection
    String folderpath = "../../../ASS/ISE(ASS)/Dataset/*.jpg";
    std::vector<String> filenames;
    cv::glob(folderpath, filenames);

    for (size_t i = 0; i < filenames.size(); i++)
    {
        Mat LPRImage = imread(filenames[i]);
        Mat GreyImage = RGBtoGrey(LPRImage);
        Mat EqualImg = EqualizeHist(GreyImage);
        /*imshow("equal", EqualImg);*/
        Mat avgImg = AverageFunction(GreyImage, 1);
        /*Mat Hor = HorizontalSobel(avgImg, 80);
        Mat VerSobel = VerticalSobel(Hor, 80);*/
        Mat Lapacion = EdgeFunction(avgImg, 255);
        
        
        Mat binarized = GreytoBinary(Lapacion, OTSU(Lapacion));
        imshow("binal", binarized);
        Mat Erosionimg = Erosion(binarized, 1);
        Mat Dilationimg = Dilation(Erosionimg, 10);
        imshow("dilated", Dilationimg);
        
       

        //Calling the segmentation lib function
        Mat blob;
        blob = Dilationimg.clone();
        std::vector<std::vector<Point>> segments;
        std::vector<Vec4i> hierarchy1;
        findContours(Dilationimg, segments, hierarchy1, RETR_EXTERNAL,
            CHAIN_APPROX_NONE, Point(0, 0)); // CCL

        Mat dst = Mat::zeros(Dilationimg.size(), CV_8UC3);

        if (!segments.empty()) {
            for (int i = 0; i < segments.size(); i++) {
                Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
                drawContours(dst, segments, i, colour, -1, 8, hierarchy1);
            }
        }
        imshow("Grey img", GreyImage);
        imshow("Segmented img", dst);

        Rect BlobRect;
        Mat Plate;
        Scalar black = CV_RGB(0, 0, 0);
        for (int j = 0; j < segments.size(); j++) {
            BlobRect = boundingRect(segments[j]);
            // if the segment at border can filter out
            // if the bounadry box is more like a square can remove it
            float ratio = (float)BlobRect.height / (float)BlobRect.width;
            if (ratio > 0.4 || BlobRect.y < GreyImage.rows * 0.09)
            {
                drawContours(blob, segments, j, black, -1, 8, hierarchy1);
            }
            else {
                Plate = GreyImage(BlobRect);
            }

        }
        //imshow("filtered", blob);

        if (Plate.rows != 0 || Plate.cols != 0)
            imshow("final Plate", Plate);

            waitKey();
    }

    /*Mat HoriSobel = HorizontalSobel(GreyImage, 50);
    Mat Dilationimg2 = Dilation(VerSobel, 5);
    Mat avgImg = AverageFunction(GreyImage, 1)
    Mat VerSobel = VerticalSobel(avgImg);
    Mat Erosionimg = Erosion(VerSobel, 1);
    Mat Dilationimg = Dilation(Erosionimg, 6);*/

}

