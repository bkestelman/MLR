#include "ANNLog.h"
#include "ANNParams.h" /* TODO: remove include, move ANN::Params to its own class */
#include<iostream>
#include<fstream>
#include<chrono>
#include<ctime>

ANNLog::ANNLog(ANNParams& params) :
	_params(params),
	functionNames()
{
	/* TODO: move to external file? */
	//std::cout << &params.activationFunc << "\n";
	Vector_t(*sigmoid)(const Vector_t&) = &MLMath::sigmoid; /* sigmoid is overloaded, so precise cast is required */
	//std::cout << sigmoid << "\n";
	/* Assign string to each function */
	functionNames[(void(*)())sigmoid] = "sigmoid"; 
	functionNames[(void(*)())&MLMath::sigmoid_d] = "sigmoid_d";
	functionNames[(void(*)())&MLMath::noFunc] = "noFunc";
	functionNames[(void(*)())&MLMath::step] = "step";
	functionNames[(void(*)())&MLMath::hebbs_d] = "hebbs_d";
	functionNames[(void(*)())&MLMath::zeroMatrix] = "zeroMatrix";
	functionNames[(void(*)())&MLMath::randMatrix] = "randMatrix";
	//functionNames[(void(*)())&ANN::adjustBiases] = "standard";
	//functionNames[(void(*)())&ANN::noBiases] = "no biases";
	/* TODO: typedef the cast to void(*)() */
}


ANNLog::~ANNLog()
{
}

std::ostream& operator<<(std::ostream& os, ANNLog& log) { /* TODO: why can't log be const? maybe map operator[] is not const... */
	os << "Learning Rate: " << log._params._learningRate << "\n";
	os << "Activation Function: " << log.functionNames[(void(*)())log._params.activationFunc] << "\n";
	os << "Activation Function Derivative: " << log.functionNames[(void(*)())log._params.activationFunc_d] << "\n";
	os << "Matrix initialization: " << log.functionNames[(void(*)())log._params._initMatrix] << "\n";
	os << "Batch size: " << log._params._batchSize << "\n";
	os << "Iterations per batch: " << log._params._iterations << "\n";
	return os;
}

void ANNLog::log(std::string file, int correct, int tests, int trains) { /* TODO: use ANNResults struct */
	std::ofstream out{ file, std::ofstream::app };
	out << "\n";
	/* TODO: clean up clock stuff */
	auto now = std::chrono::system_clock::now();
	std::time_t time = std::chrono::system_clock::to_time_t(now);
	time_t date[40]; /* TODO: is this a good value? */ 
	//ctime_s(date, 40, &time);
	ctime(date);
	out << date << "\n";
	out << "Layer sizes: { ";
	for (size_t l = 0; l < _params._layerSizes.size(); l++) {
		out << _params._layerSizes[l] << ", ";
	}
	out << "}\n";
	out << *this << "\n";
	out << _extraLog << "\n\n";
	out << "Correct: " << correct << "/" << tests << "\n";
	out << "After " << trains << " training runs " << "\n";
	out << "-------------------------------\n" << std::endl;
	out.close();
}

void ANNLog::extraLog(std::string line) {
	_extraLog += line;
}
