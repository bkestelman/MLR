#pragma once
#include "MLMath.h"

struct ANNParams { /* Set ANN parameters. Leave default when uncertain */
	using Vector_t = MLMath::Vector_t;
	using val_t = MLMath::val_t;

	val_t _stepFactor{ 1 };

	/* Activation Function and Derivatives */
	Vector_t(*activationFunc)(const Vector_t&) { &MLMath::sigmoid };
	val_t(*dn_db)(const Vector_t&, size_t) { &MLMath::noFunc_dn_db };
	val_t(*dn_dw)(const Vector_t& n, size_t node, val_t nodeBefore) { &MLMath::noFunc_dn_dw };
	val_t(*dN_dn)(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) { &MLMath::noFunc_dN_dn };
};
