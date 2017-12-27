#pragma once
#include "ANN\ANN.h"

class DataReader {
public:
	virtual ANN::Vector_t readData() = 0;
	virtual ANN::Vector_t readLabel() = 0;
};