#include "ANN.h"
#include<math.h>

/* Calculate without processing (without changing values in the ANN) */

/* Preps the next layer with weighted average plus bias calculation (no activation function) */
ANN::Vector_t ANN::prepLayerAfter(size_t nodeLayer) {
	return _weights[nodeLayer] * _layers[nodeLayer] + Vector_t::Constant(_layerSizes[nodeLayer + 1], _biases[nodeLayer]);
}

ANN::Vector_t ANN::normalize(const Vector_t& vec) {
	double tot = 0;
	for (int i = 0; i < vec.size(); i++) {
		tot += vec[i];
	}
	Vector_t normalizedVec(vec);
	for (int i = 0; i < vec.size(); i++) {
		normalizedVec[i] /= tot;
	}
	return normalizedVec;
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
ANN::val_t ANN::sigmoid_dn_db(const Vector_t& n, size_t node) {
	val_t expo = std::exp(-n[node]);
	return (expo + 1)*expo;
}
ANN::val_t ANN::sigmoid_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore) {
	val_t expo = std::exp(-n[node]);
	return (expo + 1)*expo*nodeBefore;
}
ANN::val_t ANN::sigmoid_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) {
	val_t expo = std::exp(-n[nextLayerNode]);
	return (expo + 1)*expo*weightBetween;
}


/* No activation function */
ANN::Vector_t ANN::noFunc(const Vector_t& vec) {
	return vec;
}
ANN::val_t ANN::noFunc_dn_db(const Vector_t& n, size_t node) {
	return 1;
}
ANN::val_t ANN::noFunc_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore) {
	return nodeBefore;
}
ANN::val_t ANN::noFunc_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) {
	return weightBetween;
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
