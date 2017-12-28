#include "ANN.h"
#include<math.h>
#include "Data\LogicalAND\LogicalAND.h"

/* DEPRECATED: for now, you must use your ANN with a DataReader
ANN::ANN(const std::vector<size_t>& layerSizes) :
	_layerSizes(layerSizes),
	_curLayer(0),
	_layers(_layerSizes.size()),
	_weights(_layerSizes.size() - 1),
	_biases(_layerSizes.size() - 1),
	activationFunc(&sigmoid),
	dn_db_fp(&ANN::sigmoid_dn_db),
	dn_dw_fp(&ANN::sigmoid_dn_dw),
	dN_dn_fp(&ANN::sigmoid_dN_dn),
	_dr(LogicalAND())
{
	//activationFunc = &noFunc; // TODO: is this the right place for initializing function pointers?
	for (size_t layer = 0; layer < layerSizes.size()-1; layer++) {
		_weights[layer] = Matrix_t::Zero(layerSizes[layer+1], layerSizes[layer]);
	}
}
*/

ANN::ANN(DataReader& dr) :
	_dr(dr),
	_layerSizes({dr.dataSize(), dr.labelSize()}),
	_curLayer(0), /* TODO: will getting rid of this hurt anyone? */
	_layers(_layerSizes.size()),
	_weights(_layerSizes.size() - 1),
	_biases(_layerSizes.size() - 1),
	activationFunc(&sigmoid),
	dn_db_fp(&ANN::sigmoid_dn_db),
	dn_dw_fp(&ANN::sigmoid_dn_dw),
	dN_dn_fp(&ANN::sigmoid_dN_dn)
{
	/* TODO: relocate temporary asserts */
	
	/* TODO: make an initWeights function */
	for (int layer = 0; layer < _layers.size()-1; layer++) {
		_layers[layer] = Vector_t::Zero(_layerSizes[layer]);
		_weights[layer] = Matrix_t::Zero(_layerSizes[layer + 1], _layerSizes[layer]);
		_biases[layer] = 0;
	}
	_layers[_layers.size()-1] = Vector_t::Zero(_layerSizes[_layers.size()-1]);
	_dr.testAssertions(*this);
}

ANN::Vector_t& ANN::processInput() {
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
	_layers[layer + 1] = activationFunc(prepLayerAfter(layer)); /* TODO: normalize... but where? */
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
