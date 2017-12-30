#include "MLMath.h"

/* Sigmoid */
MLMath::Vector_t MLMath::sigmoid(const Vector_t& vec) {
	Vector_t ret = vec;
	for (auto coeff = 0; coeff < vec.size(); coeff++) {
		ret[coeff] = sigmoid(vec[coeff]);
	}
	return ret;
}
MLMath::val_t MLMath::sigmoid(val_t val) {
	return 1 / (std::exp(-val) + 1);
}
MLMath::val_t MLMath::sigmoid_dn_db(const Vector_t& n, size_t node) {
	val_t expo = std::exp(-n[node]);
	return (expo + 1)*expo;
}
MLMath::val_t MLMath::sigmoid_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore) {
	val_t expo = std::exp(-n[node]);
	return (expo + 1)*expo*nodeBefore;
}
MLMath::val_t MLMath::sigmoid_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) {
	val_t expo = std::exp(-n[nextLayerNode]);
	return (expo + 1)*expo*weightBetween;
}

/* No activation function */
MLMath::Vector_t MLMath::noFunc(const Vector_t& vec) {
	return vec;
}
MLMath::val_t MLMath::noFunc_dn_db(const Vector_t& n, size_t node) {
	return 1;
}
MLMath::val_t MLMath::noFunc_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore) {
	return nodeBefore;
}
MLMath::val_t MLMath::noFunc_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) {
	return weightBetween;
}
