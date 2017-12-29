#pragma once
#include<Eigen\Dense>

class ANN;
class DataReader {
public:
	using Vector_t = Eigen::VectorXd;

	virtual Vector_t readData() = 0;
	virtual Vector_t readLabel() = 0;
	virtual const size_t dataSize() = 0;
	virtual const size_t labelSize() = 0;
	virtual const void testAssertions(const ANN&) = 0;
};