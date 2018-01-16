#pragma once
#include "Data/BufferedDataReader.h"

class StockData : public BufferedDataReader {
public:
	using Vector_t = Eigen::VectorXd;

	StockData(size_t dataSize);
	//Vector_t readData() override;
	//Vector_t readLabel() override;
	const size_t dataSize() override;
	const size_t labelSize() override;
	void seek(int) override;
	bool test(Vector_t) override;
	std::string log() override;
private:
	size_t _dataSize, _dataOffset;
	Vector_t _label;

	Vector_t readDataFromSource() override;
	Vector_t readLabelFromSource() override;

};
