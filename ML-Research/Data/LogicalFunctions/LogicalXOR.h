#pragma once
#include "Data/DataReader.h"

class LogicalXOR : public DataReader { /* TODO: maybe extend LogicalFunction */
public:
	LogicalXOR();
	DataReader::Vector_t readData() override;
	DataReader::Vector_t readLabel() override;
	const size_t dataSize() override;
	const size_t labelSize() override;
	void useTrainSet() override; /* may be implemented with seek */
	void useTestSet() override; /* may be implemented with seek */
	//const void testAssertions(const ANN& ann) override;
	bool test(Vector_t) override;
	std::string log() override;
	void init() override {};
private:
	DataReader::Vector_t data;
	DataReader::Vector_t label;
	void generateDataXORLabel();

};
