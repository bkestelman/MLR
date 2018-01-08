#include "LogicalXOR.h"

LogicalXOR::LogicalXOR() :
	data(2),
	label(1)
{
}

void LogicalXOR::generateDataXORLabel() {
	int a = rand() % 2;
	int b = rand() % 2;
	data[0] = a;
	data[1] = b;
	label[0] = a ^ b; /* XOR */
	if (label[0] == 1) assert((a == 1 && b == 0) || (a == 0 && b == 1));
	else assert((a == 0 && b == 0) || (a == 1 && b == 1));
}

DataReader::Vector_t LogicalXOR::readData() {
	generateDataXORLabel();
	return data;
}

DataReader::Vector_t LogicalXOR::readLabel() {
	return label;
}

const size_t LogicalXOR::dataSize() {
	return 2;
}

const size_t LogicalXOR::labelSize() {
	return 1;
}

bool LogicalXOR::test(DataReader::Vector_t output) {
	assert(output.size() == 1);
	return abs(output[0] - label[0]) < 0.5;
}

std::string LogicalXOR::log() { /* TODO: use operator<< */
	return "DataReader: LogicalXOR";
}