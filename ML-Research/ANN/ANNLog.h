#pragma once
#include "MLMath.h"
#include "ANN.h"
#include "ANNParams.h"
#include<map>

class ANNLog
{
	using Vector_t = MLMath::Vector_t;
	using val_t = MLMath::val_t;
public:
	ANNLog(ANN&);

	void log(std::string file);
	void extraLog(std::string line);
private:
	ANN& _ann;
	ANNParams& _params;
	std::map<void(*)(), std::string> functionNames;
	std::string _extraLog;

	friend std::ostream& operator<<(std::ostream&, ANNLog&);
};

