// DetectPlates.h

#ifndef DETECTPLATES_H
#define DETECTPLATES_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "PossiblePlate.h"
#include "PossibleChar.h"
#include "Preprocess.h"
#include "DetectChars.h"

// global constants ///////////////////////////////////////////////////////////////////////////////
const double PLATE_WIDTH_PADDING_FACTOR = 1.5;
const double PLATE_HEIGHT_PADDING_FACTOR = 1.65;

// external global variables //////////////////////////////////////////////////////////////////////
extern bool blnShowSteps;

// function prototypes ////////////////////////////////////////////////////////////////////////////
std::vector<PossiblePlate> detectPlatesInScene(cv::Mat imgOriginalScene);

std::vector<PossibleChar> findPossibleCharsInScene(cv::Mat imgThresh);

PossiblePlate extractPlate(cv::Mat imgOriginal, std::vector<PossibleChar> vectorOfMatchingChars);


# endif	// DETECTPLATES_H


