#pragma once
#include "MLMath.h"
#include<vector>

struct ANNParams { /* Set ANN parameters. Leave default when uncertain */
	using Matrix_t = MLMath::Matrix_t;
	using Vector_t = MLMath::Vector_t;
	using val_t = MLMath::val_t;

	ANNParams(std::vector<size_t>& layerSizes) :_layerSizes(layerSizes) {};

	val_t _learningRate{ 1 };
	std::vector<size_t>& _layerSizes;
	int _batchSize{ 1 };
	int _iterations{ 1 };
	val_t _beta{ 1 };

	/* Activation Function and Derivatives */
	Vector_t(*activationFunc)(const Vector_t&) { &MLMath::sigmoid };
	val_t(*activationFunc_d)(val_t) { &MLMath::sigmoid_d };

	Matrix_t(*_initMatrix)(int rows, int cols) { &MLMath::zeroMatrix };
};
