#pragma once
#include<Eigen/Dense>

class ANN;
class DataReader {
public:
	using Vector_t = Eigen::VectorXd;

	virtual Vector_t readData() = 0; /* DEPRECATED */ 
	virtual Vector_t readLabel() = 0; /* DEPRECATED */ 
	virtual const size_t dataSize() = 0;
	virtual const size_t labelSize() = 0;
	virtual void useTrainSet() = 0; /* may be implemented with seek */
	virtual void useTestSet() = 0; /* may be implemented with seek */
	//virtual void seek(int) = 0; /* No longer mandatory */
	//virtual const void testAssertions(const ANN&) = 0;
	//virtual friend std::ostream& operator<<(std::ostream&, DataReader&) = 0;
	virtual bool test(Vector_t) = 0;
	virtual std::string log() = 0;
	virtual void init() = 0;
};
