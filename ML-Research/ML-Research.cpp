// ML-Research.cpp : Defines the entry point for the console application.
//

// #include "stdafx.h"
#include<iostream>
#include<vector>
#include "ANN/ANN.h"
#include "ANN/ANNLog.h"
#include "Data/MNISTReader/MNISTReader.h"
#include "Data/LogicalFunctions/LogicalAND.h"
#include "Data/LogicalFunctions/LogicalXOR.h"
#include "Data/StockData/StockData.h"
#include "MLMath.h"
#include "StockAlgorithms.h"

/* TODO: 
 * clean up backprop
 * batch training with epochs
 * performance, references everywhere
 * softmax activation function
 * cross entropy loss function (add loss function param)
 * move out of debug build?
 * consolidate stuff in ANNCalculations, VectorCalculations, MLMath
 *
 * HIGH: 
 * hypothesis: training ann using current strategy is effective, but only for the near future 
 * however, testing on the next 10 data vectors is not enough data to confirm 
 * so... test on next 10, then make new ann starting from some offset, repeat, get statistics
 */

int main()
{
	std::cout << "Hello World" << std::endl;
	MNISTReader mr(2);
	LogicalXOR xr = LogicalXOR();
	StockData S("AMZN");
	S._dataSize = 30;
	S._predictionDistance = 30;
	DataReader& dr = mr;
	dr.init();

	std::vector<size_t> layerSizes({ dr.dataSize(), dr.labelSize() }); /* TODO: shove DataReader info
									      into params through constructor*/
	ANNParams params(layerSizes);
	params.activationFunc = MLMath::sigmoid;
	params.activationFunc_d = MLMath::sigmoid_d;
	params._learningRate = 1;
	params._initMatrix = MLMath::randMatrix;
	params._batchSize = 10000;
	params._iterations = 1;
	for(int i = 0; i < 1; i++) {
		ANN ann(dr, params);
		ann.insertLayer(1, 15);
		ann.init();
		ann.trainBatch(1);
		std::cout << "Testing\n";
		//ann.testOnBatch();
		ann.test(1000);
		ANNLog log{ann};
		log.log("ann.log");
	}

	//StockAlgorithms::momentumTest(S, 100);

	return 0;
}
