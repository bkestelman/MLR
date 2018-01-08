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
	val_t sigmoid_d(val_t val);
	val_t sigmoid_dn_db(const Vector_t& nodes, size_t node); /* all the dn_db's are deprecated */
	val_t sigmoid_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore);
	val_t sigmoid_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween);
	/* NoFunc */
	Vector_t noFunc(const Vector_t& vec);
	val_t hebbs_d(val_t x);
	val_t noFunc_dn_db(const Vector_t& n, size_t node);
	val_t noFunc_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore);
	val_t noFunc_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween);
	/* Step */
	Vector_t step(const Vector_t& vec);
	val_t step_dn_dw(const Vector_t& n, size_t node, val_t nodeBefore);
	//val_t step_dN_dn(const Vector_t& n, size_t nextLayerNode, val_t weightBetween);
	/* xorFunc */

	Matrix_t zeroMatrix(int rows, int cols);
	Matrix_t randMatrix(int rows, int cols);
};

