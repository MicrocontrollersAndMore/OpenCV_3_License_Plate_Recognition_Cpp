// Main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "DetectPlates.h"
#include "PossiblePlate.h"
#include "DetectChars.h"

#include<iostream>

// global variables ///////////////////////////////////////////////////////////////////////////////
bool blnShowSteps = false;

// global constants ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);

// function prototypes ////////////////////////////////////////////////////////////////////////////
void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
	bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();

	if (blnKNNTrainingSuccessful == false) {
		std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
		return(0);
	}

	cv::Mat imgOriginalScene;		// input image

	imgOriginalScene = cv::imread("1.png");			// open image

	if (imgOriginalScene.empty()) {									// if unable to open image
		std::cout << "error: image not read from file\n\n";		// show error message on command line
		return(0);												// and exit program
	}

	std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);

	vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);

	cv::imshow("imgOriginalScene", imgOriginalScene);			// show original image

	if (vectorOfPossiblePlates.empty()) {
		std::cout << std::endl << "no license plates were detected" << std::endl;
	} else {
				// if we get in here vector of possible plates has at leat one plate

				// sort the vector of possible plates in DESCENDING order (most number of chars to least number of chars)
		std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);

				// suppose the plate with the most recognized chars (the first plate in sorted by string length descending order)
				// is the actual plate
		PossiblePlate licPlate = vectorOfPossiblePlates.front();

		cv::imshow("imgPlate", licPlate.imgPlate);
		cv::imshow("imgThresh", licPlate.imgThresh);

		if (licPlate.strChars.length() == 0) {
			std::cout << std::endl << "no characters were detected" << std::endl << std::endl;
			return(0);
		}

		drawRedRectangleAroundPlate(imgOriginalScene, licPlate);

		std::cout << std::endl << "license plate read from image = " << licPlate.strChars << std::endl;
		std::cout << std::endl << "-----------------------------------------" << std::endl;

		writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);

		cv::imshow("imgOriginalScene", imgOriginalScene);

		cv::imwrite("imgOriginalScene.png", imgOriginalScene);
	}

	cv::waitKey(0);					// hold windows open until user presses a key

	return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
	cv::Point2f p2fRectPoints[4];

	licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);

	for (int i = 0; i < 4; i++) {
		cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
	cv::Point ptCenterOfTextArea;
	cv::Point ptLowerLeftTextOrigin;

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (double)licPlate.imgPlate.rows / 30.0;
	int intFontThickness = (int)std::round(dblFontScale * 1.5);
	int intBaseline = 0;
	
	cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, &intBaseline);

	ptCenterOfTextArea.x = (int)licPlate.rrLocationOfPlateInScene.center.x;
	
	if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {
		ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) + (int)std::round((double)licPlate.imgPlate.rows * 1.6);
	} else {
		ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) - (int)std::round((double)licPlate.imgPlate.rows * 1.6);
	}

	ptLowerLeftTextOrigin.x = (int)(ptCenterOfTextArea.x - (textSize.width / 2));
	ptLowerLeftTextOrigin.y = (int)(ptCenterOfTextArea.y + (textSize.height / 2));

	cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
}





