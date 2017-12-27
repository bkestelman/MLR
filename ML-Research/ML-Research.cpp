// ML-Research.cpp : Defines the entry point for the console application.
//

// #include "stdafx.h"
#include<iostream>
#include<vector>
#include "ANN.h"

int main()
{
	std::cout << "Hello World" << std::endl;
	size_t inputSize = 2;
	std::vector<size_t> layerSizes = { inputSize, 2, 1 };
	ANN::Vector_t input(inputSize);
	input << 1, 1;
	ANN::Vector_t label(1);
	label << 1;
	ANN ann{ layerSizes };
	std::cout << ann << std::endl;
	std::cout << "output: " << ann.processInput(input) << std::endl;
	std::cout << ann << std::endl;
	int tests = 10000;
	std::vector<ANN::Vector_t> inputs;
	ANN::Vector_t vec00(inputSize);
	ANN::Vector_t vec01(inputSize);
	ANN::Vector_t vec10(inputSize);
	ANN::Vector_t vec11(inputSize);
	vec00 << 0, 0;
	vec01 << 0, 1;
	vec10 << 1, 0;
	vec11 << 1, 1;
	inputs.push_back(vec00);
	inputs.push_back(vec01);
	inputs.push_back(vec10);
	inputs.push_back(vec11);
	std::cout << "All inputs:\n" << inputs[0] << ";\n" << inputs[1] << ";\n" << inputs[2] << ";\n" << inputs[3] << std::endl;
	std::vector<ANN::Vector_t> labels;
	ANN::Vector_t label00(1);
	ANN::Vector_t label01(1);
	ANN::Vector_t label10(1);
	ANN::Vector_t label11(1);
	label00 << 0;
	label01 << 0;
	label10 << 0;
	label11 << 1;
	labels.push_back(label00);
	labels.push_back(label01);
	labels.push_back(label10);
	labels.push_back(label11);
	std::cout << "backprop testing" << std::endl;
	for (int i = 0; i < tests; i++) {
		std::cout << "Input:\n" << inputs[i % 4] << "\nEnd input" << std::endl;
		std::cout << "Output before training:\n" << ann.train(inputs[i % 4], labels[i % 4]) << "\nEnd output" << std::endl;
		std::cout << "Output after training:\n" << ann.processInput(inputs[i % 4]) << "\nEndoutput" << std::endl;
		std::cout << ann << std::endl;
	}
    return 0;
}

