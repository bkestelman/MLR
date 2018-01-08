#include "MLMath.h"

static double beta = 1; /* TODO: move */

/* Sigmoid */
MLMath::Vector_t MLMath::sigmoid(const Vector_t& vec) {
	Vector_t ret = vec;
	for (auto coeff = 0; coeff < vec.size(); coeff++) {
		ret[coeff] = sigmoid(vec[coeff]);
	}
	return ret;
}
MLMath::val_t MLMath::sigmoid(val_t val) {
	return 1 / (std::exp(-beta*val) + 1);
}
MLMath::val_t MLMath::sigmoid_d(val_t x) {
	val_t expo = std::exp(-x);
	return (1 / ((expo + 1)*(expo + 1))) * expo;
}
MLMath::val_t MLMath::sigmoid_dn_db(const Vector_t& n, size_t node) {
	val_t expo = std::exp(-beta*n[node]);
	return (1 / ((expo + 1)*(expo + 1))) * expo;
}
MLMath::val_t MLMath::sigmoid_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore) {
	val_t expo = std::exp(-beta*n[node]);
	return (1 / ((expo + 1)*(expo + 1))) * expo * beta * nodeBefore;
}
MLMath::val_t MLMath::sigmoid_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) {
	val_t expo = std::exp(-beta*n[nextLayerNode]);
	return (1 / ((expo + 1)*(expo + 1))) * expo * beta * weightBetween;
}

/* Step Function */
MLMath::Vector_t MLMath::step(const Vector_t& vec) {
	Vector_t ret = vec;
	for (int i = 0; i < vec.size(); i++) {
		vec[i] > 0 ? ret[i] = 1 : ret[i] = 0;
	}
	return ret;
}
MLMath::val_t MLMath::step_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore) {
	return nodeBefore; /* Hebb's rule; derivative of weightedSum, not of step, with respect to weight */
}

/* No activation function */
MLMath::Vector_t MLMath::noFunc(const Vector_t& vec) {
	return vec;
}
MLMath::val_t MLMath::hebbs_d(val_t x) {
	return 1;
}
MLMath::val_t MLMath::noFunc_dn_db(const Vector_t& n, size_t node) {
	return 1;
}
MLMath::val_t MLMath::noFunc_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore) {
	return nodeBefore;
}
MLMath::val_t MLMath::noFunc_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) {
	// N = W*n
	// dN_dn (derivative of Next node with respect to node) = dN_dw + dw_dn = 
	return weightBetween;
}

MLMath::Matrix_t MLMath::zeroMatrix(int rows, int cols) {
	return Matrix_t::Zero(rows, cols);
}
MLMath::Matrix_t MLMath::randMatrix(int rows, int cols) {
	return Matrix_t::Random(rows, cols);
}