#include "ANNLog.h"
#include "ANNParams.h" /* TODO: remove include, move ANN::Params to its own class */
#include<iostream>
#include<fstream>
#include<chrono>
#include<ctime>

ANNLog::ANNLog(ANNParams params) :
	_params(params),
	functionNames()
{
	/* TODO: move to external file? */
	//std::cout << &params.activationFunc << "\n";
	Vector_t(*sigmoid)(const Vector_t&) = &MLMath::sigmoid; /* sigmoid is overloaded, so precise cast is required */
	//std::cout << sigmoid << "\n";
	/* Assign string to each function */
	functionNames[(void(*)())sigmoid] = "sigmoid"; 
	functionNames[(void(*)())&MLMath::noFunc] = "noFunc";
	functionNames[(void(*)())&MLMath::sigmoid_dn_dw] = "sigmoid_dn_dw";
	functionNames[(void(*)())&MLMath::sigmoid_dn_db] = "sigmoid_dn_db";
	functionNames[(void(*)())&MLMath::sigmoid_dN_dn] = "sigmoid_dN_dn";
	functionNames[(void(*)())&MLMath::noFunc_dn_dw] = "noFunc_dn_dw";
	functionNames[(void(*)())&MLMath::noFunc_dn_db] = "noFunc_dn_db";
	functionNames[(void(*)())&MLMath::noFunc_dN_dn] = "noFunc_dN_dn";
	/* TODO: typedef the cast to void(*)() */
}


ANNLog::~ANNLog()
{
}

std::ostream& operator<<(std::ostream& os, ANNLog& log) { /* TODO: why can't log be const? maybe map operator[] is not const... */
	os << "Activation Function: " << log.functionNames[(void(*)())log._params.activationFunc] << "\n";
	os << "dn_dw: " << log.functionNames[(void(*)())log._params.dn_dw] << "\n";
	os << "dn_db: " << log.functionNames[(void(*)())log._params.dn_db] << "\n";
	os << "dN_dn: " << log.functionNames[(void(*)())log._params.dN_dn] << std::endl;
	return os;
}

void ANNLog::log(std::string file, int correct, int tests, int trains) { /* TODO: use ANNResults struct */
	std::ofstream out{ file, std::ofstream::app };
	out << "\n";
	/* TODO: clean up clock stuff */
	auto now = std::chrono::system_clock::now();
	std::time_t time = std::chrono::system_clock::to_time_t(now);
	char date[40]; /* TODO: is this a good value? */
	ctime_s(date, 40, &time);
	out << date << "\n";
	out << *this << std::endl;
	out << "Correct: " << correct << "/" << tests << "\n";
	out << "After " << trains << " training runs " << "\n";
	out << "-------------------------------\n" << std::endl;
	out.close();
}