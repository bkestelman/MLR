// ML-Research.cpp : Defines the entry point for the console application.
//

// #include "stdafx.h"
#include<iostream>
#include<vector>
#include "ANN\ANN.h"
#include "Data\MNISTReader\MNISTReader.h"
#include "Data\LogicalAND\LogicalAND.h"
#include "SystemInfo.h"
#include "VectorCalculations.h"
#include "MLMath.h"

int main()
{
	std::cout << "Hello World" << std::endl;
	DataReader& dr = MNISTReader();
	ANNParams params;
	params.activationFunc = MLMath::sigmoid;
	params.dn_dw = MLMath::noFunc_dn_dw;
	params.dn_db = MLMath::noFunc_dn_db;
	params.dN_dn = MLMath::noFunc_dN_dn;
	ANN ann(dr, params);
	//ann.insertLayer(10);
	int train = 100;
	int test = 10;
	for (int i = 0; i < train; i++) {
		std::cout << "i: " << i << std::endl;
		if (i >= train - test) {
			ann.test();
		}
		else ann.train();
	}
	//std::cout << ann << std::endl;
	ann.log("ann.log");
	//std::cout << ann._log << std::endl;
	/*
	size_t inputSize = 784;
	size_t outputSize = 10;
	std::vector<size_t> layerSizes = { inputSize, outputSize };
	ANN::Vector_t input(inputSize);
	ANN::Vector_t label(outputSize);
	ANN ann{ layerSizes };
	int tests = 1000;
	DataReader& dr = MNISTReader();
	std::cout << "backprop testing" << std::endl;
	//dr.readLabel();
	for (int i = 0; i < tests; i++) {
		input = dr.readData();
		label = dr.readLabel();
		assert(label.size() == outputSize);
		//std::cout << "Input:\n" << input << "\nEnd input" << std::endl;
		if (tests - i < 10) {
			std::cout << "Label:\n" << label << "\nEnd label" << std::endl;
			std::cout << "Output before training:\n" << ann.train(input, label) << "\nEnd output" << std::endl;
			std::cout << "Output after training:\n" << ann.processInput(input) << "\nEndoutput" << std::endl;
		}
		//std::cout << ann << std::endl;
	}*/
    return 0;
}
