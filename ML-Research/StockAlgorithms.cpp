#include "StockAlgorithms.h"
#include<iostream>

std::ofstream StockAlgorithms::_log("stock_alg.log");

void StockAlgorithms::momentumTest(StockData& S, int tests) {
	int correct = 0;
	for(int i = 0; i < tests; i++) {
		std::cout << "[StockAlgorithms momentumTest()] i: " << i << "\n";
		Vector_t data = S.readData();	
		double delta = data[data.size()-1] > data[0]; /* 1 if price rose, 0 if price fell */
		if(delta == S.readLabel()[0]) correct++;
	}	
	_log << "Correct: " << correct << "/" << tests << "\n";
}
