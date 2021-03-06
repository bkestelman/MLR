#pragma once
#include "Data/DataReader.h"

class ANN;
class LogicalAND : public DataReader {
public:
	LogicalAND();
	DataReader::Vector_t readData() override;
	DataReader::Vector_t readLabel() override;
	const size_t dataSize() override;
	const size_t labelSize() override;
	//const void testAssertions(const ANN& ann) override;
	bool test(Vector_t) override;
	std::string log() override;
private:
	DataReader::Vector_t data;
	DataReader::Vector_t label;
	void generateDataAndLabel();

};
