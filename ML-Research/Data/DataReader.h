#pragma once
#include<Eigen/Dense>

class ANN;
class DataReader {
public:
	using Vector_t = Eigen::VectorXd;

	virtual Vector_t readData() = 0; 
	virtual Vector_t readLabel() = 0;
	virtual const size_t dataSize() = 0;
	virtual const size_t labelSize() = 0;
	virtual void prepareBuffer() = 0;
	virtual void seek(int) = 0;
	//virtual const void testAssertions(const ANN&) = 0;
	//virtual friend std::ostream& operator<<(std::ostream&, DataReader&) = 0;
	virtual bool test(Vector_t) = 0;
	virtual std::string log() = 0;
};
