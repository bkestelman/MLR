// ML-Research.cpp : Defines the entry point for the console application.
//

// #include "stdafx.h"
#include<iostream>
#include<vector>
#include "ANN\ANN.h"
#include "Data\LogicalAND\LogicalAND.h"

int main()
{
	std::cout << "Hello World" << std::endl;
	size_t inputSize = 2;
	std::vector<size_t> layerSizes = { inputSize, 1 };
	ANN::Vector_t input(inputSize);
	input << 1, 1;
	ANN::Vector_t label(1);
	label << 1;
	ANN ann{ layerSizes };
	std::cout << ann << std::endl;
	std::cout << "output: " << ann.processInput(input) << std::endl;
	std::cout << ann << std::endl;
	int tests = 1000;
	LogicalAND and;
	std::cout << "backprop testing" << std::endl;
	for (int i = 0; i < tests; i++) {
		input = and.readData();
		label = and.readLabel();
		std::cout << "Input:\n" << input << "\nEnd input" << std::endl;
		std::cout << "Output before training:\n" << ann.train(input, label) << "\nEnd output" << std::endl;
		std::cout << "Output after training:\n" << ann.processInput(input) << "\nEndoutput" << std::endl;
		std::cout << ann << std::endl;
	}
    return 0;
}

