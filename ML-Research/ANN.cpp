#include "ANN.h"
#include<math.h>

ANN::ANN(const std::vector<size_t>& layerSizes)
	:_layerSizes(layerSizes),
	_curLayer(0),
	_layers(_layerSizes.size()),
	_weights(_layerSizes.size() - 1),
	_biases(_layerSizes.size() - 1)
{
	activationFunc = &noFunc; // TODO: is this the right place for initializing function pointers?
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
	_layers[layer + 1] = activationFunc(prepLayerAfter(layer));
}
