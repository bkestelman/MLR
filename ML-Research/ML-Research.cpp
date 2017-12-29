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

int main()
{
	std::cout << "Hello World" << std::endl;
	DataReader& dr = MNISTReader();
	ANN ann(dr);
	//ann.insertLayer(10);
	int tests = 100;
	int correct = 0;
	for (int i = 0; i < tests; i++) {
		//std::cout << ann << "End ANN" << std::endl;
		std::cout << "i: " << i << std::endl;
		ann.readNext();
		int labelIndex = maxIndex(ann.label());
		Eigen::VectorXd output = ann.train();
		int outputIndex = maxIndex(output);
		if (labelIndex == outputIndex) correct++;
		if (i >= tests - 10) {
			//std::cout << "Input:\n" << ann.inputs() << "\nEndInputs\n";
			//std::cout << ann.inputs();
			std::cout << "Label: " << labelIndex << "\n" << ann.label() << "\nEndLabel\n";
			std::cout << "Output before training: " << outputIndex << "\n" << output << "\nEndTrain\n\n";
			std::cout << "Output after training:\n" << ann.processInput() << "\nEndTrain\n" << std::endl;
		}
		else ann.train(); /* train without printing */
	}
	//std::cout << ann << std::endl;
	std::cout << "Correct: " << correct << std::endl;
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
