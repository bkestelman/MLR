#pragma once
#include "Data/StockData/StockData.h"
#include<fstream>

class StockAlgorithms {
	using Vector_t = Eigen::VectorXd; /* TODO: some way to do this without referring to Eigen? */

public:
	static std::ofstream _log;
	/*
	 * Attempts to predict future prices by simply following 
	 * momentum trend of known prices. i.e. if stock rose in 
	 * past 30 days, predict it will rise over next 30 days.
	 */
	static void momentumTest(StockData& S, int tests);
};
