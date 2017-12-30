#include "ANN.h"
#include<math.h>

/* Calculate without processing (without changing values in the ANN) */

/* Preps the next layer with weighted average plus bias calculation (no activation function) */
ANN::Vector_t ANN::prepLayerAfter(size_t nodeLayer) {
	return _weights[nodeLayer] * _layers[nodeLayer] + Vector_t::Constant(_layerSizes[nodeLayer + 1], _biases[nodeLayer]);
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
	assert(false); 
	assert(vec.size() > 0);
	double range = 0;
	double max = vec[0];
	double min = vec[0];
	for (auto i = 0; i < vec.size(); i++) {
		if (vec[i] > max) max = vec[i];
		else if (vec[i] < min) min = vec[i];
	}
	if (max == min) return;
	Vector_t normalizedVec(vec);
	for (auto i = 0; i < vec.size(); i++) {
		vec[i] /= (max - min);
	}
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
