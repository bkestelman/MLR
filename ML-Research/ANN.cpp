#include "ANN.h"
#include<math.h>

ANN::ANN(const std::vector<size_t>& layerSizes) 
	:_layerSizes(layerSizes), 
	_curLayer(0),
	_layers(_layerSizes.size()),
	_weights(_layerSizes.size()-1), 
	_biases(_layerSizes.size()-1)
{
	for (size_t layer = 0; layer < layerSizes.size()-1; layer++) {
		_weights[layer] = Matrix_t::Zero(layerSizes[layer+1], layerSizes[layer]);
	}
}

ANN::Vector_t& ANN::processInput(const Vector_t& input) {
	setInput(input);
	for (size_t layer = 0; layer < _layerSizes.size() - 1; layer++) {
		processLayer(layer);
	}
	return _layers[_layerSizes.size()-1];
}

void ANN::setInput(const Vector_t& input) {
	_layers[0] = input;
}

void ANN::processLayer(size_t layer) {
	_layers[layer + 1] = _weights[layer] * _layers[layer] + Vector_t::Constant(_layerSizes[layer + 1], _biases[layer]);
	_layers[layer + 1] = sigmoid(_layers[layer + 1]);
}

ANN::Vector_t ANN::sigmoid(const Vector_t& vec) {
	Vector_t ret = vec;
	for (auto coeff = 0; coeff < vec.size(); coeff++) {
		ret[coeff] = 1 / (std::exp(-vec[coeff]) + 1);
	}
	return ret;
}
