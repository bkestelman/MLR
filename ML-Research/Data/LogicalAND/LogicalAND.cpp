#include "LogicalAND.h"

LogicalAND::LogicalAND() :
	data(2),
	label(1)
{
}

void LogicalAND::generateDataAndLabel() {
	data[0] = rand() % 2;
	data[1] = rand() % 2;
	if (data[0] == 1 && data[1] == 1) label[0] = 1;
	else label[0] = 0;
}

ANN::Vector_t LogicalAND::readData() {
	generateDataAndLabel();
	return data;
}

ANN::Vector_t LogicalAND::readLabel() {
	return label;
}