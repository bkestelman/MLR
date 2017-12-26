#include "ANN.h"

ANN::Vector_t& ANN::train(const Vector_t& input, const Vector_t& label) {
	assert(_layers[outputLayer()].size() == label.size());
	backprop(processInput(input), label);
	return _layers[outputLayer()];
}

void ANN::backprop(const Vector_t& output, const Vector_t& label) {
	Vector_t output_dE_dn = dE_dn(output, label); /* Error vector */
	for (int weightLayer = (int)lastWeightLayer(); weightLayer >= 0; weightLayer--) { /* for loop bounds to work, weightLayer must be signed
																					  (alternatively, offset the bounds) */
		output_dE_dn = backpropLayer((size_t)weightLayer, output_dE_dn);                    /* <- but this expects weightLayer unsigned */
	}
}

/* Derivative of Square Difference Error Function with respect to each output node value (dE_dn) */
ANN::Vector_t ANN::dE_dn(const Vector_t& output, const Vector_t& label) {
	Vector_t dE_dn_vec{ output.size() };
	for (auto node = 0; node < output.size(); node++) {
		dE_dn_vec[node] = output[node] - label[node]; /* technically should be *2 */
	}
	return dE_dn_vec;
}

/* dE_dn is the backpropagated derivative of error function with respect to nodes in the nodeLayer after weightLayer */
ANN::Vector_t ANN::backpropLayer(size_t weightLayer, const Vector_t& backpropagated_dE_dn) { /* TODO: check for memory leak */	
	Vector_t new_dE_dn = backprop_dE_dn(weightLayer, backpropagated_dE_dn);	
	adjustWeights(weightLayer, backpropagated_dE_dn); 
	adjustBiases(weightLayer, backpropagated_dE_dn);
	return new_dE_dn; /* pass this to the next backPropLayer call */
}

void ANN::adjustWeights(size_t weightLayer, const Vector_t& backpropagated_dE_dn) {
	for (auto row = 0; row < _weights[weightLayer].rows(); row++) {
		for (auto col = 0; col < _weights[weightLayer].cols(); col++) {
			double dn_dw = nodeBefore(weightLayer, row, col); /* change in nodeAfter, caused by changing weight */
			double dE_dw = dn_dw * backpropagated_dE_dn[row]; /* chain rule */
			_weights[weightLayer](row, col) -= dE_dw * _stepFactor;
		}
	}
}

void ANN::adjustBiases(size_t weightLayer, const Vector_t& backpropagated_dE_dn) {
	for (auto node = 0; node < _layers[nodeLayerAfter(weightLayer)].size(); node++) {
		_biases[node] -= backpropagated_dE_dn[node] * _stepFactor;
	}
}

ANN::Vector_t ANN::backprop_dE_dn(size_t nodeLayer, const Vector_t& backpropagated_dE_dn) {
	Vector_t new_dE_dn{ _layers[nodeLayer].size() };
	for (auto node = 0; node < _layers[nodeLayer].size(); node++) {
		double dE_dn = 0; 
		for (auto nextLayerNode = 0; nextLayerNode < _layers[nodeLayer + 1].size(); nextLayerNode++) {
			dE_dn += weight(weightLayerAfter(nodeLayer), node, nextLayerNode) * backpropagated_dE_dn[nextLayerNode]; /* chain rule */
		}
		new_dE_dn[node] = dE_dn;
	}
	return new_dE_dn;
}