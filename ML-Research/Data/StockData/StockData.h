#pragma once
#include "Data/DataReader.h"

class StockData : public DataReader {
public:
	using Vector_t = Eigen::VectorXd;

	StockData(size_t dataSize);
	Vector_t readData() override;
	Vector_t readLabel() override;
	const size_t dataSize() override;
	const size_t labelSize() override;
	bool test(Vector_t) override;
	std::string log() override;
private:
	size_t _dataSize, _dataOffset;
	Vector_t _label;
};
