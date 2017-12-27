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
ANN::val_t ANN::dn_db_f(const Vector_t& n, size_t node) {
	return (*this.*dn_db_fp)(n, node);
}
ANN::val_t ANN::noFunc_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore) {
	return nodeBefore;
}
ANN::val_t ANN::dn_dw_f(const Vector_t& n, size_t node, val_t nodeBefore) {
	return (*this.*dn_dw_fp)(n, node, nodeBefore);
}
ANN::val_t ANN::noFunc_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) {
	return weightBetween;
}
ANN::val_t ANN::dN_dn_f(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) {
	return (*this.*dN_dn_fp)(n, nextLayerNode, weightBetween);
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
