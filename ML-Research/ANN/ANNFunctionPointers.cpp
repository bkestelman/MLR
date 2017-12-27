#include "ANN.h"

ANN::val_t ANN::dn_db_f(const Vector_t& n, size_t node) {
	return (*this.*dn_db_fp)(n, node);
}
ANN::val_t ANN::dn_dw_f(const Vector_t& n, size_t node, val_t nodeBefore) {
	return (*this.*dn_dw_fp)(n, node, nodeBefore);
}
ANN::val_t ANN::dN_dn_f(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) {
	return (*this.*dN_dn_fp)(n, nextLayerNode, weightBetween);
}