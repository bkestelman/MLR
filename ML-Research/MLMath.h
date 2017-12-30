#pragma once
#include<Eigen\Dense>

namespace MLMath
{
	using Matrix_t = Eigen::MatrixXd;
	using Vector_t = Eigen::VectorXd;
	using val_t = double;

	/* Sigmoid */
	Vector_t sigmoid(const Vector_t& vec);
	val_t sigmoid(val_t val);
	val_t sigmoid_dn_db(const Vector_t& nodes, size_t node);
	val_t sigmoid_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore);
	val_t sigmoid_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween);
	/* NoFunc */
	Vector_t noFunc(const Vector_t& vec);
	val_t noFunc_dn_db(const Vector_t& n, size_t node);
	val_t noFunc_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore);
	val_t noFunc_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween);
};

