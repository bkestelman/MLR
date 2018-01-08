#include "LogicalAND.h"

LogicalAND::LogicalAND() :
	data(2),
	label(1)
{
}

void LogicalAND::generateDataAndLabel() {
	int a = rand() % 2;
	int b = rand() % 2; 
	data[0] = a;
	data[1] = b;
	label[0] = a & b;
}

DataReader::Vector_t LogicalAND::readData() {
	generateDataAndLabel();
	return data;
}

DataReader::Vector_t LogicalAND::readLabel() {
	return label;
}

const size_t LogicalAND::dataSize() {
	return 2;
}

const size_t LogicalAND::labelSize() {
	return 1;
}

//const void LogicalAND::testAssertions(const ANN& ann) {
//	/* TODO: unhard code these (maybe make a function that other testAssertions also call) */
//	assert(ann._layerSizes.size() == 2);
//	assert(ann._layers.size() == 2);
//	assert(ann._layers[0].size() == 2);
//	assert(ann._layers[1].size() == 1);
//	assert(ann._weights.size() == 1);
//	assert(ann._weights[0].rows() == 1);
//	assert(ann._weights[0].cols() == 2);
//	assert(ann._biases.size() == 1);
//}

bool LogicalAND::test(DataReader::Vector_t output) {
	assert(output.size() == 1);
	return abs(output[0] - label[0]) < 0.5;
}

std::string LogicalAND::log() { /* TODO: use operator<< */
	return "DataReader: LogicalAND";
}