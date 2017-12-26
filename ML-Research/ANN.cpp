#include "ANN.h"

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

ANN::Vector_t& ANN::train(const Vector_t& input, const Vector_t& label) {
	backprop(processInput(input), label);
	return _layers[outputLayer()];
}
void ANN::backprop(const Vector_t& output, const Vector_t& label) {
	double error = scalarError(output, label);
	Vector_t dE_dn_cur = dE_dn(output, label);
	for (int weightLayer = (int)lastWeightLayer(); weightLayer >= 0; weightLayer--) { /* for loop to work, weightLayer must be signed
																					  (alternatively, offset the bounds) */
		dE_dn_cur = backpropLayer((size_t)weightLayer, dE_dn_cur); // but this expects weightLayer unsigned
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
	_layers[layer+1] = _weights[layer] * _layers[layer] + Vector_t::Constant(_layerSizes[layer+1], _biases[layer]);
}
double ANN::scalarError(const Vector_t& output, const Vector_t& label) {
	assert(output.size() == label.size());
	double error = 0;
	for (auto coeff = 0; coeff < output.size(); coeff++) {
		double diff = output(coeff) - label(coeff);
		error += diff * diff;
	}
	return error;
}
ANN::Vector_t ANN::dE_dn(const Vector_t& output, const Vector_t& label) {
	Vector_t dE_dn_vec{ output.size() };
	for (auto node = 0; node < output.size(); node++) {
		dE_dn_vec[node] = output[node] - label[node]; // technically should be *2
	}
	return dE_dn_vec;
}
/* dE_dn is the backprop derivative of error with respect to nodes in the nodeLayer after weightLayer */
ANN::Vector_t ANN::backpropLayer(size_t weightLayer, const Vector_t& dE_dn) {
	// derivs with respect to weights
	Matrix_t new_weights(_weights[weightLayer]); // copy of weights
	for (auto row = 0; row < _weights[weightLayer].rows(); row++) {
		for (auto col = 0; col < _weights[weightLayer].cols(); col++) {
			double dn_dw = nodeBefore(weightLayer, row, col); // change in nodeAfter, caused by changing weight
			double dE_dw = dn_dw * dE_dn[row]; // chain rule
			new_weights(row, col) -= dE_dw * _stepFactor;
		}
	}
	for (auto node = 0; node < _layers[nodeLayerAfter(weightLayer)].size(); node++) {
		_biases[node] -= dE_dn[node] * _stepFactor;
	}
	// prepare dE_dn for prev layer
	Vector_t beforeWeights_dE_dn{ _layers[nodeLayerBefore(weightLayer)].size() };
	for (auto nodeBeforeWeights = 0; nodeBeforeWeights < _layers[nodeLayerBefore(weightLayer)].size(); nodeBeforeWeights++) {
		double dE_dpn = 0; // 
		for (auto nodeAfterWeights = 0; nodeAfterWeights < _layers[nodeLayerAfter(weightLayer)].size(); nodeAfterWeights++) {
			dE_dpn += weight(weightLayer, nodeBeforeWeights, nodeAfterWeights) * dE_dn[nodeAfterWeights];
		}
		beforeWeights_dE_dn[nodeBeforeWeights] = dE_dpn;
	}	
	_weights[weightLayer] = new_weights;
	return beforeWeights_dE_dn;
}




ANN::val_t ANN::nodeBefore(size_t weightLayer, size_t row, size_t col) {
	return _layers[weightLayer](col);
}
ANN::val_t ANN::nodeAfter(size_t weightLayer, size_t row, size_t col) {
	return _layers[weightLayer + 1](row);
}
size_t ANN::nodeLayerAfter(size_t weightLayer) {
	return weightLayer + 1;
}
size_t ANN::nodeLayerBefore(size_t weightLayer) {
	return weightLayer;
}
ANN::val_t ANN::weight(size_t weightLayer, size_t nodeBefore, size_t nodeAfter) {
	return _weights[weightLayer](nodeAfter, nodeBefore);
}
size_t ANN::lastWeightLayer() {
	return _weights.size() - 1;
}
size_t ANN::outputLayer() {
	return _layers.size() - 1;
}






std::ostream& operator<<(std::ostream& os, const std::vector<ANN::Matrix_t> matrices) {
	for (auto& matrix : matrices) {
		os << matrix << "\n" << std::endl;
	}
	return os;
}
std::ostream& operator<<(std::ostream& os, const std::vector<ANN::val_t> biases) {
	for (auto& bias : biases) {
		os << bias << ", ";
	}
	os << std::endl;
	return os;
}
std::ostream& operator<<(std::ostream& os, const ANN& ann) {
	os << "ANN\n---\n";
	os << "Weights\n";
	os << ann._weights;
	os << "Biases\n";
	os << ann._biases;
	return os;
}
