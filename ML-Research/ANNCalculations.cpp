#include "ANN.h"
#include<math.h>

/* Calculate without processing (without changing values in the ANN) */

/* Preps the next layer with weighted average plus bias calculation (no activation function) */
ANN::Vector_t ANN::prepLayerAfter(size_t nodeLayer) {
	return _weights[nodeLayer] * _layers[nodeLayer] + Vector_t::Constant(_layerSizes[nodeLayer + 1], _biases[nodeLayer]);
}

/* Coefficient wise sigmoid on vector */
ANN::Vector_t ANN::sigmoid(const Vector_t& vec) {
	Vector_t ret = vec;
	for (auto coeff = 0; coeff < vec.size(); coeff++) {
		ret[coeff] = sigmoid(vec[coeff]);
	}
	return ret;
}
ANN::val_t ANN::sigmoid(val_t val) {
	return 1 / (std::exp(-val) + 1);
}
/* No activation function */
ANN::Vector_t ANN::noFunc(const Vector_t& vec) {
	return vec;
}

/* Derivative of Square Difference Error Function with respect to each output node value (dE_dn) */
ANN::Vector_t ANN::output_dE_dn(const Vector_t& output, const Vector_t& label) {
	Vector_t dE_dn{ output.size() };
	for (auto node = 0; node < output.size(); node++) {
		dE_dn[node] = output[node] - label[node]; /* technically should be * 2...
													  Yes, it should be output - label,
													  not label - output (calculate the derivative of a square difference function) */
	}
	return dE_dn;
}
