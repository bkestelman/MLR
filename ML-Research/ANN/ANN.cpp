#include "ANN.h"
#include "ANNLog.h"
#include<math.h>
#include "Data\LogicalAND\LogicalAND.h"
#include "VectorCalculations.h"

ANN::ANN(DataReader& dr) :
	_dr(dr),
	_layerSizes({dr.dataSize(), dr.labelSize()}),
	_curLayer(0), /* TODO: will getting rid of this hurt anyone? */
	_layers(_layerSizes.size()),
	_weights(_layerSizes.size() - 1),
	_biases(_layerSizes.size() - 1),
	_log(_params)
{
	/* TODO: make an initWeights function */
	for (int layer = 0; layer < _layers.size()-1; layer++) {
		_layers[layer] = Vector_t::Zero(_layerSizes[layer]);
		_weights[layer] = Matrix_t::Zero(_layerSizes[layer + 1], _layerSizes[layer]);
		_biases[layer] = 0;
	}
	_layers[_layers.size()-1] = Vector_t::Zero(_layerSizes[_layers.size()-1]);
	_dr.testAssertions(*this);
}

ANN::ANN(DataReader& dr, ANNParams& params) :
	_dr(dr),
	_layerSizes({ dr.dataSize(), dr.labelSize() }),
	_curLayer(0), /* TODO: will getting rid of this hurt anyone? */
	_layers(_layerSizes.size()),
	_weights(_layerSizes.size() - 1),
	_biases(_layerSizes.size() - 1),
	_params(params),
	_log(params)
{
	/* TODO: make an initWeights function */
	for (int layer = 0; layer < _layers.size() - 1; layer++) {
		_layers[layer] = Vector_t::Zero(_layerSizes[layer]);
		_weights[layer] = Matrix_t::Zero(_layerSizes[layer + 1], _layerSizes[layer]);
		_biases[layer] = 0;
	}
	_layers[_layers.size() - 1] = Vector_t::Zero(_layerSizes[_layers.size() - 1]);
	_dr.testAssertions(*this);
}

void ANN::test() {
	_tests++;
	readNext();
	int labelIndex = maxIndex(_label);
	Eigen::VectorXd output = processInput();
	int outputIndex = maxIndex(output);
	if (labelIndex == outputIndex) _correct++;
	std::cout << "Label: " << _label << "\n";
	std::cout << "Output: " << outputIndex << "\n" << output << std::endl;
}

ANN::Vector_t& ANN::processInput() {
	//scale(_input);
	setInput(_input);
	for (size_t layer = 0; layer < _layerSizes.size() - 1; layer++) {
		processLayer(layer);
	}
	return _layers[_layerSizes.size()-1];
}

void ANN::setInput(const Vector_t& input) {
	_layers[0] = input;
}

void ANN::processLayer(size_t layer) {
	//scale(_layers[layer]);
	_layers[layer + 1] = _params.activationFunc(prepLayerAfter(layer)); /* TODO: normalize... but where? */
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
	_label = _dr.readLabel();
}

void ANN::insertLayer(size_t size) {
	_layers.insert(_layers.begin() + 1, Vector_t::Zero(size));
	_weights.insert(_weights.begin() + 1, Matrix_t::Zero(_layers[1].size(), _layers[0].size()));
	_weights[1] = Matrix_t::Zero(_layers[2].size(), _layers[1].size());
	_biases.insert(_biases.begin() + 1, 0);
	assert(_weights.size() == 2);
	assert(_layers.size() == 3);
	for (auto i = 0; i < _weights.size(); i++) {
		assert(_weights[i].cols() == _layers[i].size());
		assert(_weights[i].rows() == _layers[i+1].size());
	}
	//assert(false);
}

void ANN::log(std::string file) {
	_log.log(file, _correct, _tests, _trains);
}

int ANN::correct() {
	return _correct;
}