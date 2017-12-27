#pragma once
#include "ANN\ANN.h"
#include "Data\DataReader.h"

class LogicalAND : DataReader {
public:
	LogicalAND::LogicalAND();
	ANN::Vector_t readData() override;
	ANN::Vector_t readLabel() override;
private:
	ANN::Vector_t data;
	ANN::Vector_t label;
	void generateDataAndLabel();
};