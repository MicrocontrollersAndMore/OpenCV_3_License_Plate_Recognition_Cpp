// PossibleChar.h

#ifndef POSSIBLE_CHAR_H
#define POSSIBLE_CHAR_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////////
class PossibleChar {
public:
    // member variables ///////////////////////////////////////////////////////////////////////////
    std::vector<cv::Point> contour;

    cv::Rect boundingRect;

    int intCenterX;
    int intCenterY;

    double dblDiagonalSize;
    double dblAspectRatio;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortCharsLeftToRight(const PossibleChar &pcLeft, const PossibleChar & pcRight) {
        return(pcLeft.intCenterX < pcRight.intCenterX);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    bool operator == (const PossibleChar& otherPossibleChar) const {
        if (this->contour == otherPossibleChar.contour) return true;
        else return false;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    bool operator != (const PossibleChar& otherPossibleChar) const {
        if (this->contour != otherPossibleChar.contour) return true;
        else return false;
    }

    // function prototypes ////////////////////////////////////////////////////////////////////////
    PossibleChar(std::vector<cv::Point> _contour);

};

#endif  // POSSIBLE_CHAR_H


