// ISE(Lab2).cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cmath>
#include "highgui/highgui.hpp"
#include "core.hpp"
#include "opencv2/opencv.hpp"

// test OCR
#include <baseapi.h>
#include <allheaders.h>

using namespace cv;
// code an RGB to GREY

Mat RGBtoGrey(Mat RGB) {
	Mat Grey = Mat::zeros(RGB.size(), CV_8UC1);
	// 8 bits one channel, so compiler knows, pixel only one value
	// changng CV_8UC3 then it turns tousing 3 columns like RGB
	for (int i = 0; i < RGB.rows; i++) {
		for (int j = 0; j < RGB.cols * 3; j = j + 3) {
			Grey.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3;

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
			else {
				Binary.at<uchar>(i, j) = 0;
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
			else if (sum > th) {
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

			gx = (Grey.at<uchar>(i, j - 1) * -2)
				+ (Grey.at<uchar>(i - 1, j - 1) * -1)
				+ (Grey.at<uchar>(i + 1, j - 1) * -1)
				+ (Grey.at<uchar>(i - 1, j + 1) * 1)
				+ (Grey.at<uchar>(i, j + 1) * 2)
				+ (Grey.at<uchar>(i + 1, j + 1) * 1);

			if (abs(gx) > th) {
				outputImg.at<uchar>(i, j) = 255;
			}
			/*else {
				outputImg.at<uchar>(i, j) = 255;
			}*/

			//int LeftSide = -1 * Grey.at<uchar>(i - 1, j - 1) - 2 * Grey.at<uchar>(i, j - 1) - 1 * Grey.at<uchar>(i + 1, j - 1);
			//int RightSide = Grey.at<uchar>(i - 1, j + 1) + 2 * Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i + 1, j + 1);
			//if (abs(LeftSide + RightSide) > 80) //50
			//	outputImg.at<uchar>(i, j) = 255;

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

// first detection algorithm
Mat firstSetDetection(Mat grey) {
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(7, 1), Point(-1, -1));
	Mat BlackHat;
	Mat GradientMorph;
	Mat Closing;

	Mat EqualImg = EqualizeHist(grey);
	Mat avgImg = AverageFunction(EqualImg, 1);
	//enhance dark object in bright background
	morphologyEx(EqualImg, BlackHat, MORPH_BLACKHAT, element);
	// highlighting the structure of img
	morphologyEx(BlackHat, GradientMorph, MORPH_GRADIENT, element2);
	morphologyEx(GradientMorph, Closing, MORPH_CLOSE, element2);
	Mat binarized = GreytoBinary(Closing, OTSU(Closing));
	Mat erosionimg = Erosion(binarized, 1);
	Mat output = Dilation(erosionimg, 3);

	return output;
}

// second detection algorithm
Mat secondSetDetection(Mat grey) {
	Mat EQImg = EqualizeHist(grey);
	Mat BlurImg = AverageFunction(EQImg, 1);
	Mat VEdge = VerticalSobel(BlurImg, 80);
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat dilated;
	morphologyEx(VEdge, dilated, MORPH_DILATE, element);

	//filter the noises around after first morphology
	// to prevent collision with other non-plate segment
	// when further dilation
	Mat blob;
	blob = dilated.clone();
	std::vector<std::vector<Point>> segments;
	std::vector<Vec4i> hierarchy1;
	findContours(dilated, segments, hierarchy1, RETR_EXTERNAL,
		CHAIN_APPROX_NONE, Point(0, 0)); // CCL

	Rect BlobRect;
	Mat Plate;
	Scalar black = CV_RGB(0, 0, 0);
	for (int j = 0; j < segments.size(); j++) {
		// filter big noises
		BlobRect = boundingRect(segments[j]);
		float area = (float)BlobRect.height * (float)BlobRect.width;
		if (
			BlobRect.height > 40
			|| BlobRect.width > 160
			|| area > 6300
			|| area < 500
			)
		{
			drawContours(blob, segments, j, black, -1, 8, hierarchy1);
		}
	}

	/* dilate with mask focused on 6 horizontally
	* 3 vertically
	* which the result will have more dilation horizontally
	* causing the segments will combine more horizontally
	*/
	Mat element1 = getStructuringElement(MORPH_RECT, Size(6, 3), Point(-1, -1));
	Mat dilated1;
	morphologyEx(blob, dilated1, MORPH_DILATE, element1);

	return dilated1;
}

int countWhitePixel(Mat grey) {
	int count = 0;
	// Start from second row / column
	// Stop before the last row / column
	for (int i = 1; i < grey.rows; i++) {
		for (int j = 1; j < grey.cols; j++) {
			if (grey.at<uchar>(i, j) == 255) {
				count++;
			}
		}
	}
	return count;
}

// code for final character recognition
int recogniseCha(Mat finalPlate) {

	// invert font from white to black, for tesseract accuracy
	finalPlate = Invert(finalPlate);

	// Add some border for more accurate tesseract
	int top, left, bottom, right;
	top = (int)(0.2 * finalPlate.rows); bottom = top;
	left = (int)(0.2 * finalPlate.cols); right = left;
	copyMakeBorder(finalPlate, finalPlate, top, bottom, left, right, BORDER_CONSTANT, 255);

	char* outText;
	tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
	// Initialize tesseract-ocr with English, without specifying tessdata path
	if (api->Init("C:\\Users\\Admin\\Desktop\\OCR\\tessdata", "eng")) {
		fprintf(stderr, "Could not initialize tesseract.\n");
		exit(1);
	}

	api->SetImage((uchar*)finalPlate.data, finalPlate.size().width, finalPlate.size().height, finalPlate.channels(), finalPlate.step1());

	// Open input image with leptonica library
	// Get OCR result
	outText = api->GetUTF8Text();

	int count = 0;
	for (int i = 0; outText[i] != '\0'; ++i) {
		if (outText[i] != '\n') {
			// count num of output
			if (outText[i] != ' ') {
				count++;
			}
			// remove puntuations
			if (ispunct(outText[i])) {
				memmove(outText + i, outText + i + 1, strlen(outText) - i);
			}
		}
	}

	// remove if first character is empty
	if (isspace(outText[0])) {
		for (int i = 0; outText[i] != '\0'; ++i) {
			outText[i] = outText[i + 1];
		}

	}

	if (count >= 5) {
		printf("OCR output:\n%s", outText);
		imshow("final to feed", finalPlate);
		waitKey();
	}

	// Destroy used object and release memory
	api->End();
	delete api;
	delete[] outText;
	return count;
}

// count how many white grouped pixels passed
// in 3 lines
int countCrossCut(Mat grey) {
	int count1 = 0;
	int count2 = 0;
	int count3 = 0;

	// Start from second column
	int firstCut = grey.rows / 3;
	int secondCut = grey.rows / 2;
	int thirdCut = grey.rows * 2 / 3;

	for (int j = 1; j < grey.cols; j++) {
		if (
			grey.at<uchar>(firstCut, j) == 255
			&& grey.at<uchar>(firstCut, j - 1) == 0
			) {
			count1++;
		}
	}

	for (int j = 1; j < grey.cols; j++) {
		if (
			grey.at<uchar>(secondCut, j) == 255
			&& grey.at<uchar>(secondCut, j - 1) == 0
			) {
			count2++;
		}
	}

	for (int j = 1; j < grey.cols; j++) {
		if (
			grey.at<uchar>(thirdCut, j) == 255
			&& grey.at<uchar>(thirdCut, j - 1) == 0
			) {
			count3++;
		}
	}

	return count1 + count2 + count3;
}

int main()
{
	// Loop every image in dataset for license plate detection
	String folderpath = "../../../ASS/ISE(ASS)/Dataset/*.jpg";
	std::vector<String> filenames;
	cv::glob(folderpath, filenames);

	for (size_t i = 0; i < filenames.size(); i++)
	{
		Mat detectedPlate;
		Mat LPRImage = imread(filenames[i]);
		Mat GreyImage = RGBtoGrey(LPRImage);

		// Looping for 2 sets of detection algo
		for (int l = 0; l < 2; l++)
		{
			if (l == 0) {
				//use first set detection
				detectedPlate = firstSetDetection(GreyImage);
			}
			else {
				//use second set detection
				detectedPlate = secondSetDetection(GreyImage);
			}

			//Calling the segmentation lib function
			Mat blob;
			blob = detectedPlate.clone();
			std::vector<std::vector<Point>> segments;
			std::vector<Vec4i> hierarchy1;
			findContours(detectedPlate, segments, hierarchy1, RETR_EXTERNAL,
				CHAIN_APPROX_NONE, Point(0, 0)); // CCL

			std::cout << "Start of img *********************************************************" << std::endl;
			Rect BlobRect;
			Mat filteredSegment;
			Scalar black = CV_RGB(0, 0, 0);

			// used for selecting highest cross cut img
			std::vector<int> vect;
			vect.push_back(0);
			vect.push_back(0);

			for (int j = 0; j < segments.size(); j++) {
				BlobRect = boundingRect(segments[j]);
				float ratio = (float)BlobRect.height / (float)BlobRect.width;
				float area = (float)BlobRect.height * (float)BlobRect.width;
				float compactness = contourArea(segments[j]) / area;

				// filter based on ration, position, height, width, area and compactness
				if (
					(ratio < 0.9 && ratio > 0.1)
					&& BlobRect.y > GreyImage.rows * 0.2
					&& BlobRect.y < GreyImage.rows * 0.9
					&& BlobRect.height < 65
					&& BlobRect.width > 47
					&& (area > 1000 && area < 6500)
					&& compactness > 0.6
					)
				{
					filteredSegment = GreyImage(BlobRect);

					// run calculate cross cut for every candidates
					// find the highest cross cut
					if (vect[0] < countCrossCut(GreytoBinary(filteredSegment, OTSU(filteredSegment + 20)))) {
						vect[0] = countCrossCut(GreytoBinary(filteredSegment, OTSU(filteredSegment + 20)));
						vect[1] = j;
					}

				}
				else {
					//filtered out all the noises
					drawContours(blob, segments, j, black, -1, 8, hierarchy1);
				}

			}

			Mat Plate;
			float fontThickness;
			Rect BlobRect1;

			// 0 means no segment found
			if (vect[1] != 0) {
				// assign the highest cross cut segment into rect
				BlobRect1 = boundingRect(segments[vect[1]]);

				// Draw Rectangle on the image
				rectangle(LPRImage, BlobRect1.tl(), BlobRect1.br(), CV_RGB(255, 0, 0), 2);
				imshow("plate location", LPRImage);
				waitKey();

				/*
				* extend some pixels on left and right of the segment,
				* to ensure all character are included
				*/
				BlobRect1.x -= 5;
				BlobRect1.width += 10;

				//binarised final plate and calculate approximately the font thickness
				Plate = GreytoBinary(GreyImage(BlobRect1), OTSU(GreyImage(BlobRect1) + 50));
				fontThickness = (float)countWhitePixel(Plate) / (Plate.rows * Plate.cols);
			}
			else {
				Plate = NULL;
			}

			// if not empty plate
			if ((Plate.rows != 0 || Plate.cols != 0) && countWhitePixel(Plate) != 0) {
				//miss like 4 or 5? best so far
				/*int startingOtsu = abs((fontThickness * fontThickness * fontThickness) * 2400);*/

				/*
				* calculate starting otsu based on font thickness
				* thicker font, higher otsu value,
				* for better tesseract recognition
				*/
				int startingOtsu = abs((fontThickness * fontThickness) * 800 + 5);

				// Loop adding otsu value if result invalid
				for (int i = startingOtsu; i < 255; i = i + 40) {
					Plate = GreytoBinary(GreyImage(BlobRect1), OTSU(GreyImage(BlobRect1) + i));

					//Calling the segmentation lib function
					Mat blob;
					blob = Plate.clone();
					std::vector<std::vector<Point>> ChaSegments;
					std::vector<Vec4i> hierarchy2;
					findContours(Plate, ChaSegments, hierarchy2, RETR_EXTERNAL,
						CHAIN_APPROX_NONE, Point(0, 0)); // CCL

					Mat denoisedPlate = Plate.clone();
					Rect BlobRect1;
					// filter out noises in plate
					for (int k = 0; k < ChaSegments.size(); k++) {
						BlobRect1 = boundingRect(ChaSegments[k]);
						int area = BlobRect1.height * BlobRect1.width;
						if (BlobRect1.height < 10
							|| BlobRect1.width < 1
							|| BlobRect1.width > 95
							|| area > 5000
							) {
							drawContours(denoisedPlate, ChaSegments, k, black, -1, 8, hierarchy2);
						}
						/*imshow("denoised plate", character);
						waitKey();*/
					}

					/*
					* if output is more than 5 num and letters
					* break from the looping of otsu value
					*/
					if (recogniseCha(denoisedPlate) >= 5) {
						break;
					}
				}

				/*
				* this break is for breaking the detectiong set algorithm
				* after the plate is found
				* if this is not breaked
				* it will run the second set of detection algorithm
				*/
				break;
			}
		}
	}
}

