// ML-Research.cpp : Defines the entry point for the console application.
//

// #include "stdafx.h"
#include<iostream>
#include<vector>
#include "ANN/ANN.h"
#include "Data/MNISTReader/MNISTReader.h"
#include "Data/LogicalFunctions/LogicalAND.h"
#include "Data/LogicalFunctions/LogicalXOR.h"
#include "MLMath.h"

/* TODO: 
- clean up backprop
- batch training with epochs
- performance, references everywhere
- softmax activation function
- cross entropy loss function (add loss function param)
- move out of debug build?
- test on linux
- consolidate stuff in ANNCalculations, VectorCalculations, MLMath
*/

int main()
{
	std::cout << "Hello World" << std::endl;
	//std::ofstream os{ "out.log" };
	MNISTReader mr = MNISTReader();
	mr._dataStep = 2; 
	mr._scaleDown = 255;
	DataReader& dr = mr;
	std::vector<size_t> layerSizes({ dr.dataSize(), dr.labelSize() }); /* TODO: shove DataReader info
																	   into params through constructor*/
	ANNParams params(layerSizes);
	params.activationFunc = MLMath::sigmoid;
	params.activationFunc_d = MLMath::sigmoid_d;
	/*params.activationFunc = MLMath::step;
	params.activationFunc_d = MLMath::hebbs_d;*/
	/*params.dn_dw = MLMath::step_dn_dw;
	params.dn_db = MLMath::noFunc_dn_db;
	params.dN_dn = MLMath::noFunc_dN_dn;*/
	params._learningRate = 1;
	params._initMatrix = MLMath::randMatrix;
	params._batchSize = 1;
	params._iterations = 1;
	ANN ann(dr, params);
	ann.insertLayer(1, 15);
	//ann.insertLayer(1, 4);
	int train = 1000;
	int test = 1000;
	//MLMath::Vector_t output;
	//std::cout << ann << "\n";
	/*ann._weights[0] = ANN::Matrix_t(2, 2);
	ann._weights[0] << 3, 3, -3, -3;
	ann._biases[0] = ANN::Vector_t(2);
	ann._biases[0] << -1, 4;
	ann._weights[1] = ANN::Matrix_t(1, 2);
	ann._weights[1] << 1, 1;
	ann._biases[1] = ANN::Vector_t(1);
	ann._biases[1] << -1;
	std::cout << ann << "\n";*/
	//for (int i = 0; i < test; i++) { // pre-train tests
	//	ann.test(); /* TODO: return test output */
	//}
	for (int i = 0; i < train + test; i++) {
		std::cout << "i: " << i << "\n";
		//os << "i: " << i << "\n";
		//output = ann.processInput();
		if (i >= train) {
			ann.test(); /* TODO: return test output */
		}
		else ann.train();
		//os << "Label:\n" << ann.label() << "\n" << "End Label" << "\n";
		//os << "Output before training:\n" << output << "\n" << "End output" << "\n";
		//os << "layer 1:\n" << ann.layer(1) << "\n";
		//output = ann.processInput();
		//os << "Output after training:\n" << output << "\n" << "End output" << "\n";
		//os << "layer 1:\n" << ann.layer(1) << "\n";
		//os << "Biases: " << ann._biases[0] << "\n";// << ", " << ann._biases[1] << "\n";
		//os << "Labelcount:\n" << mr._labelCounts << "\n";
		//std::cout << ann << std::endl;
	}
	//std::cout << ann << std::endl;
	ann.log("ann.log");
	//os.close();
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
