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
//	_dr.testAssertions(*this);
}

void ANN::init() {
	initLayers();
	initWeights();
	_input = _layers[0]; 
}
void ANN::initLayers() {
	for(size_t i = 0; i < _layerSizes.size(); i++) {
		bool biasNode = (i != outputLayer()); 
		_layers.push_back(Vector_t::Constant(_layerSizes[i]+biasNode, -1)); /* No bias node on outputLayer */
	}
}
void ANN::initWeights() {
	for(size_t weightLayer = 0; weightLayer < _layerSizes.size()-1; weightLayer++) {
		_weights.push_back(initWeightMatrix(weightLayer));
	}
}
ANN::Matrix_t ANN::initWeightMatrix(size_t weightLayer) {
	return Matrix_t::Random(sizeNoBias(nodeLayerAfter(weightLayer)), sizeWithBias(nodeLayerBefore(weightLayer)));
}

ANN::Vector_t& ANN::processInput() {
	setInput();
	for(size_t i = 0; i < outputLayer(); i++) {
		processLayer(i);
	}
	return _layers[outputLayer()];
}

ANN::Vector_t& ANN::test() {
	_tests++;
	readNext();
	Eigen::VectorXd output = processInput();	
	_trainLog << "\nLabel:\n" << _label << "\nOutput:\n" << output << "\n";
	if (_dr.test(output)) {
		_correct++;
		_trainLog << "Correct\n";
		_trainLog << "count: " << _correct << "\n";
	}
	return _layers[outputLayer()];
}

void ANN::setInput() {
	_layers[0].head(sizeNoBias(0)) = _input; /* TODO: make function for setting layer, without touching bias node */
}

void ANN::processLayer(size_t layer) { /* aka feedforward */
	_layers[layer + 1].head(sizeNoBias(layer+1)) = _params.activationFunc(weightedSum(layer)); /* process next layer (but not its bias node) */
}

ANN::Vector_t& ANN::inputs() {
	return _input;
}

ANN::Vector_t& ANN::label() {
	return _label;
}

void ANN::readNext() {
	_input = _dr.readData();
	_label = _dr.readLabel();
}

void ANN::insertLayer(size_t layer, size_t size) {
	assert(layer < _layerSizes.size() && layer > 0); /* cannot insert new input/output layer */
	_layerSizes.insert(_layerSizes.begin() + layer, size);
	//setupLayer(layer);
}
void ANN::setupLayer(size_t layer) { /* DEPRECATED */
	assert(layer < _layerSizes.size());
	if(layer < _layerSizes.size() - 1) 
		_layers.insert(_layers.begin() + layer, Vector_t::Constant(_layerSizes[layer] + 1, -1)); /* extra bias node fixed -1 (arbitrary) */
	else 
		_layers.insert(_layers.begin() + layer, Vector_t::Constant(_layerSizes[layer], -1)); /* no extra bias node for output layer */
	if (_layers.size() == 1) return; // if inserted layer is only layer, no weights to init
	if (layer != 0) { // if not first layer, init weights before layer
		insertWeightsBefore(layer);
	}
	if (layer != _layers.size() - 1) {
		setWeightsAfter(layer);
	}
}
void ANN::insertWeightsBefore(size_t layer) { /* also inserts bias */ /* DEPRECATED */
	assert(layer != 0);
	_weights.insert(_weights.begin() + weightLayerBefore(layer), 
		_params._initMatrix(_layers[layer].size() - 1, _layers[layer - 1].size())); /* one less row (no weights to next layer bias node) */
	_weightsDeltas.insert(_weightsDeltas.begin() + weightLayerBefore(layer), 
		Matrix_t::Zero(_layers[layer].size() - 1, _layers[layer - 1].size()));
}
void ANN::setWeightsAfter(size_t layer) { /* also inserts bias */ /* DEPRECATED */
	assert(layer != _layers.size() - 1);
	bool weightsToOutput = (layer == outputLayer()-1);
	_weights[layer] = _params._initMatrix(_layers[layer + 1].size() - !weightsToOutput, _layers[layer].size()); /* one less row (no weights to next layer bias node) */
	_weightsDeltas[layer] = Matrix_t::Zero(_layers[layer + 1].size() - !weightsToOutput, _layers[layer].size());
}


void ANN::log(std::string file) {
	_log.extraLog(_dr.log());
	_log.log(file, _correct, _tests, _trains); /* TODO: use results struct */
}

int ANN::correct() {
	return _correct;
}
