#pragma once
#include "MLMath.h"
#include "ANNParams.h"
#include<map>

class ANNLog
{
	using Vector_t = MLMath::Vector_t;
	using val_t = MLMath::val_t;
public:
	ANNLog(ANNParams);
	~ANNLog();

	void log(std::string file, int correct, int tests, int trains);
private:
	ANNParams _params;
	std::map<void(*)(), std::string> functionNames;

	friend std::ostream& operator<<(std::ostream&, ANNLog&);
};

