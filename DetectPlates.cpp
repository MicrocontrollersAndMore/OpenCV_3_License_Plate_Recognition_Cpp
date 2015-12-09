// DetectPlates.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "DetectPlates.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<PossiblePlate> detectPlatesInScene(cv::Mat imgOriginalScene) {
	
	std::vector<PossiblePlate> vectorOfPossiblePlates;			// this will be the return value

	cv::Mat imgGrayscaleScene;
	cv::Mat imgThreshScene;
	cv::Mat imgContours(imgOriginalScene.size(), CV_8UC1, cv::Scalar(0.0));

	cv::destroyAllWindows();

	if (blnShowSteps) {
		cv::imshow("0", imgOriginalScene);
	}

	preprocess(imgOriginalScene, imgGrayscaleScene, imgThreshScene);

	if (blnShowSteps) {
		cv::imshow("1a", imgGrayscaleScene);
		cv::imshow("1b", imgThreshScene);
	}

	std::vector<PossibleChar> vectorOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene);
	
	if (blnShowSteps) {
		
		std::vector<std::vector<cv::Point> > contours;

		for (auto possibleChar = begin(vectorOfPossibleCharsInScene); possibleChar != end(vectorOfPossibleCharsInScene); possibleChar++) {
			contours.push_back(possibleChar->contour);
		}
		cv::drawContours(imgContours, contours, -1, cv::Scalar(255.0));
		cv::imshow("2b", imgContours);
	}

	std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInScene = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInScene);

	if (blnShowSteps) {
		cv::Mat imgContours(imgOriginalScene.size(), CV_8UC1, cv::Scalar(0.0));

		for (auto vectorOfMatchingChars = vectorOfVectorsOfMatchingCharsInScene.begin(); vectorOfMatchingChars != vectorOfVectorsOfMatchingCharsInScene.end(); vectorOfMatchingChars++) {
			cv::RNG rng;
			int intRandomBlue = rng.uniform(0, 256);
			int intRandomGreen = rng.uniform(0, 256);
			int intRandomRed = rng.uniform(0, 256);

			for (auto matchingChar = vectorOfMatchingChars->begin(); matchingChar != vectorOfMatchingChars->end(); matchingChar++) {
				cv::drawContours(imgContours, matchingChar->contour, 0, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
			}
		}
		cv::imshow("3", imgContours);
	}

	for (auto vectorOfMatchingChars = vectorOfVectorsOfMatchingCharsInScene.begin(); vectorOfMatchingChars != vectorOfVectorsOfMatchingCharsInScene.end(); vectorOfMatchingChars++) {
		PossiblePlate possiblePlate = extractPlate(imgOriginalScene, *vectorOfMatchingChars);

		if (possiblePlate.imgPlate.empty() == false) {
			vectorOfPossiblePlates.push_back(possiblePlate);
		}
	}

	std::cout << std::endl << vectorOfPossiblePlates.size() << " possible plates found" << std::endl;

	if (blnShowSteps) {
		std::cout << std::endl;
		cv::imshow("4a", imgContours);

		for (unsigned int i = 0; i < vectorOfPossiblePlates.size(); i++) {
			cv::Point2f p2fRectPoints[4];

			vectorOfPossiblePlates[i].rrLocationOfPlateInScene.points(p2fRectPoints);

			for (int j = 0; j < 4; j++) {
				cv::line(imgContours, p2fRectPoints[j], p2fRectPoints[(j + 1) % 4], cv::Scalar(0.0, 0.0, 255.0), 2);
			}
			cv::imshow("4a", imgContours);

			std::cout << "possible plate " << i << ", click on any image and press a key to continue . . ." << std::endl;

			cv::imshow("4b", vectorOfPossiblePlates[i].imgPlate);
			cv::waitKey(0);
		}
		std::cout << std::endl << "plate detection complete, click on any image and press a key to begin char recognition . . ." << std::endl << std::endl;
		cv::waitKey(0);
	}
	return vectorOfPossiblePlates;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<PossibleChar> findPossibleCharsInScene(cv::Mat imgThresh) {
	std::vector<PossibleChar> vectorOfPossibleChars;			// this will be the return value
	
	cv::Mat imgContours(imgThresh.size(), CV_8UC1, cv::Scalar(0.0));
	int intCountOfValidPossibleChars = 0;

	cv::Mat imgThreshCopy = imgThresh.clone();

	std::vector<std::vector<cv::Point> > contours;

	cv::findContours(imgThreshCopy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	for (unsigned int i = 0; i < contours.size(); i++) {

		if (blnShowSteps) {
			cv::drawContours(imgContours, contours, i, cv::Scalar(255.0));
		}

		std::vector<cv::Point> contour;

		cv::approxPolyDP(cv::Mat(contours[i]), contour, cv::arcLength(cv::Mat(contours[i]), true) * 0.0001, true);

		PossibleChar possibleChar(contour);

		if (checkIfPossibleChar(possibleChar)) {
			intCountOfValidPossibleChars++;
			vectorOfPossibleChars.push_back(possibleChar);
		}

	}

	if (blnShowSteps) {
		std::cout << std::endl << "contours.size() = " << contours.size() << std::endl;								// 2115 with MCLRNF1 image
		std::cout << "step 2 - intCountOfValidPossibleChars = " << intCountOfValidPossibleChars << std::endl;		// 174 with MCLRNF1 image
		cv::imshow("2a", imgContours);
	}

	return(vectorOfPossibleChars);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
PossiblePlate extractPlate(cv::Mat imgOriginal, std::vector<PossibleChar> vectorOfMatchingChars) {
	PossiblePlate possiblePlate;			// this will be the return value

							// sort chars from left to right based on x position
	std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

	double dblPlateCenterX = (double)(vectorOfMatchingChars[0].intCenterX + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterX) / 2.0;
	double dblPlateCenterY = (double)(vectorOfMatchingChars[0].intCenterY + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY) / 2.0;

	cv::Point2d p2dPlateCenter(dblPlateCenterX, dblPlateCenterY);

	int intPlateWidth = (int)((vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.x + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.width - vectorOfMatchingChars[0].boundingRect.x) * PLATE_WIDTH_PADDING_FACTOR);
	
	double intTotalOfCharHeights = 0;

	for (auto matchingChar = vectorOfMatchingChars.begin(); matchingChar != vectorOfMatchingChars.end(); matchingChar++) {
		intTotalOfCharHeights = intTotalOfCharHeights + matchingChar->boundingRect.height;
	}
	
	double dblAverageCharHeight = (double)intTotalOfCharHeights / vectorOfMatchingChars.size();
	
	int intPlateHeight = (int)(dblAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR);

	double dblOpposite = vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY - vectorOfMatchingChars[0].intCenterY;
	double dblHypotenuse = distanceBetweenChars(vectorOfMatchingChars[0], vectorOfMatchingChars[vectorOfMatchingChars.size() - 1]);
	double dblCorrectionAngleInRad = asin(dblOpposite / dblHypotenuse);
	double dblCorrectionAngleInDeg = dblCorrectionAngleInRad * (180.0 / 3.14159);

	cv::Mat rotationMatrix = cv::getRotationMatrix2D(p2dPlateCenter, dblCorrectionAngleInDeg, 1.0);

	cv::warpAffine(possiblePlate.imgPlate, possiblePlate.imgPlate, rotationMatrix, possiblePlate.imgPlate.size());

	return(possiblePlate);
}







