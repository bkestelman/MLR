#include "ANN.h"
#include<math.h>
#include "VectorCalculations.h"

ANN::ANN(DataReader& dr, ANNParams& params) :
	_dr(dr),
	_layers(),
	_weights(),
	_params(params),
	_layerSizes(_params._layerSizes),
	_log(_params)
{
	for (int layer = 0; layer < _layerSizes.size(); layer++) {
		setupLayer(layer);
	}
//	_dr.testAssertions(*this);
	_input = _layers[0];

	_inputBatch = std::vector<Vector_t>(_params._batchSize);
	_labelBatch = std::vector<Vector_t>(_params._batchSize);
}

ANN::Vector_t& ANN::test() {
	_tests++;
	readNext();
	Eigen::VectorXd output = processInput(0);	
	_trainLog << "\nLabel:\n" << _label << "\nOutput:\n" << output << "\n";
	if (_dr.test(output)) {
		_correct++;
		_trainLog << "Correct\n";
		_trainLog << "count: " << _correct << "\n";
	}
	return _layers[outputLayer()];
}

ANN::Vector_t ANN::processInput(int input) {
	setInput(_inputBatch[input]);
	for (size_t layer = 0; layer < _layers.size() - 1; layer++) {
		processLayer(layer);
	}
	return _layers[outputLayer()].head(_layers[outputLayer()].size() - 1); /* don't return bias node */
}

void ANN::setInput(const Vector_t& input) {
	_layers[0].head(_layers[0].size()-1) = input; /* TODO: make function for setting layer, without touching bias node */
}

void ANN::processLayer(size_t layer) { /* aka feedforward */
	//scale(_layers[layer]);
	_layers[layer + 1].head(_layers[layer + 1].size() - 1) = _params.activationFunc(weightedSum(layer)); /* process next layer (but not its bias node) */
	//scale(_layers[layer + 1]);
}

ANN::Vector_t& ANN::inputs() {
	return _input;
}

ANN::Vector_t& ANN::label() {
	return _label;
}

void ANN::readNext() {
	_input = _dr.readData();
	_inputBatch[0] = _input;
	//normalize(_input);
	_label = _dr.readLabel();
}
void ANN::readBatch() {
	for (int i = 0; i < _params._batchSize; i++) {
		readNext();
		//normalize(_input);
		_inputBatch[i] = _input;
		_labelBatch[i] = _label;
	}
}

void ANN::insertLayer(size_t layer, size_t size) {
	assert(layer <= _layers.size());
	_layerSizes.insert(_layerSizes.begin() + layer, size);
	setupLayer(layer);
}
void ANN::setupLayer(size_t layer) {
	_layers.insert(_layers.begin() + layer, Vector_t::Constant(_layerSizes[layer] + 1, -1)); /* extra bias node fixed -1 (arbitrary) */
	//assert(_layers[0].size() == 3);
	if (_layers.size() == 1) return; // if inserted layer is only layer, no weights to init
	if (layer != 0) { // if not first layer, init weights before layer
		insertWeightsBefore(layer);
	}
	if (layer != _layers.size() - 1) {
		setWeightsAfter(layer);
	}
	for (auto weightLayer = 0; weightLayer < _weights.size(); weightLayer++) {
		assert(_weights[weightLayer].cols() == _weightsDeltas[weightLayer].cols());
		assert(_weights[weightLayer].rows() == _weightsDeltas[weightLayer].rows());
		assert(_weights[weightLayer].cols() == _layers[nodeLayerBefore(weightLayer)].size());
		assert(_weights[weightLayer].rows() == _layers[nodeLayerAfter(weightLayer)].size() - 1); /* one less row (no weights to next layer bias node) */
	}
	for (auto nodeLayer = 0; nodeLayer < _layers.size() - 1; nodeLayer++) {
		assert(_layers[nodeLayer].size() == _weights[weightLayerAfter(nodeLayer)].cols());
		assert(_layers[nodeLayer].size() == 1 + _layerSizes[nodeLayer]);
	}
}
void ANN::insertWeightsBefore(size_t layer) { /* also inserts bias */
	assert(layer != 0);
	_weights.insert(_weights.begin() + weightLayerBefore(layer), 
		_params._initMatrix(_layers[layer].size() - 1, _layers[layer - 1].size())); /* one less row (no weights to next layer bias node) */
	_weightsDeltas.insert(_weightsDeltas.begin() + weightLayerBefore(layer), 
		Matrix_t::Zero(_layers[layer].size() - 1, _layers[layer - 1].size()));
}
void ANN::setWeightsAfter(size_t layer) { /* also inserts bias */
	assert(layer != _layers.size() - 1);
	_weights[layer] = _params._initMatrix(_layers[layer + 1].size() - 1, _layers[layer].size()); /* one less row (no weights to next layer bias node) */
	_weightsDeltas[layer] = Matrix_t::Zero(_layers[layer + 1].size() - 1, _layers[layer].size());
}


void ANN::log(std::string file) {
	_log.extraLog(_dr.log());
	_log.log(file, _correct, _tests, _trains); /* TODO: use results struct */
}

int ANN::correct() {
	return _correct;
}