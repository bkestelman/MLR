#include "ANN.h"
#include<math.h>

/* Calculate without processing (without changing values in the ANN) */

ANN::Vector_t ANN::weightedSum(size_t nodeLayer) const { /* TODO: test performance change returning const Vector_t& */
	return _weights[weightLayerAfter(nodeLayer)] * _layers[nodeLayer];
}

void ANN::scale(Vector_t& vec) {
	//assert(false);
	assert(vec.size() > 0);
	val_t max = vec[0];
	for (auto i = 0; i < vec.size(); i++) {
		if (vec[i] > max) max = vec[i]; /* TODO: account for negatives */
	}
	if (max == 0) return;
	for (auto i = 0; i < vec.size(); i++) {
		vec[i] /= max;
		assert(vec[i] <= 1 && vec[i] >= -1); 
	}
}

void ANN::normalize(Vector_t& vec) {
	assert(vec.size() > 0);
	double tot = 0;
	for (auto i = 0; i < vec.size(); i++) {
		tot += vec[i];
	}
	if (tot == 0) return;
	for (auto i = 0; i < vec.size(); i++) {
		vec[i] /= tot;
	}
}

/* Derivative of Square Difference Error Function with respect to each output node value (dE_dn) */
ANN::Vector_t ANN::output_dE_dn(const Vector_t& output, const Vector_t& label) {
	Vector_t dE_dn{ output.size() };
	for (auto node = 0; node < output.size(); node++) {
		dE_dn[node] = output[node] - label[node]; /* technically should be * 2...
							     Yes, it should be output - label,
							     not label - output (calculate the derivative of a square difference function
							     with respect to output) */
	}
	return dE_dn;
}

ANN::val_t ANN::error() {
	val_t err = 0;
	Vector_t output = _layers[outputLayer()];
	for (auto node = 0; node < output.size(); node++) {
		val_t diff = output[node] - _label[node]; 
		err += diff * diff; 
	}
	return err;
}
