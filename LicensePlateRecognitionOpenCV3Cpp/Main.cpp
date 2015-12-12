// Main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "DetectPlates.h"
#include "PossiblePlate.h"
#include "DetectChars.h"

#include<iostream>

// global variables ///////////////////////////////////////////////////////////////////////////////
bool blnShowSteps = true;

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {

	cv::Mat imgOriginalScene;		// input image
	
	imgOriginalScene = cv::imread("1.png");			// open image

	if (imgOriginalScene.empty()) {									// if unable to open image
		std::cout << "error: image not read from file\n\n";		// show error message on command line
		return(0);												// and exit program
	}

	std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);

	bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();

	if (blnKNNTrainingSuccessful == false) {
		std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
		return(0);
	}

	vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);

	cv::imshow("imgOriginalScene", imgOriginalScene);			// show original image

	if (vectorOfPossiblePlates.empty()) {
		//
	} else {
				// if we get in here vector of possible plates has at leat one plate

				// sort the vector of possible plates in DESCENDING order (most number of chars to least number of chars)
		std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);

				// suppose the plate with the most recognized chars (the first plate in sorted by string length descending order)
				// is the actual plate
		PossiblePlate licPlate = vectorOfPossiblePlates.front();

		cv::imshow("imgPlate", licPlate.imgPlate);
		cv::imshow("imgThresh", licPlate.imgThresh);
		cv::imwrite("imgThresh.png", licPlate.imgThresh);

		if (licPlate.strChars.length() == 0) {
			std::cout << std::endl << "no characters were detected" << std::endl << std::endl;
		}

		cv::Point2f p2fRectPoints[4];
		
		licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);

		for (int i = 0; i < 4; i++) {
			cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
		}

		std::cout << std::endl << "license plate read from image = " << licPlate.strChars << std::endl;
		std::cout << "-----------------------------------------" << std::endl;

		std::string strText = "example text";
		int intFontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
		double dblFontScale = 2;
		int intFontThickness = 3;

		cv::Point ptTextOrigin;

		cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, 0);

		ptTextOrigin.x = (imgOriginalScene.cols - textSize.width) / 2;
		ptTextOrigin.y = (imgOriginalScene.rows + textSize.height) / 2;

		cv::putText(imgOriginalScene, strText, ptTextOrigin, intFontFace, dblFontScale, cv::Scalar(0, 255, 255), intFontThickness);

	}

	cv::waitKey(0);					// hold windows open until user presses a key

	return(0);
}

