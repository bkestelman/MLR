#pragma once
#include<vector>
#include<iostream>
#include<Eigen\Dense>
#include "Data\DataReader.h"
#include "Data\LogicalAND\LogicalAND.h"
#include "Data\MNISTReader\MNISTReader.h"
#include "MLMath.h"

class ANN {
public:
	using Matrix_t = Eigen::MatrixXd;
	using Vector_t = Eigen::VectorXd;
	using val_t = double;
	using data_t = int; /* Move to DataReader or use <> (TODO: what is this?) */

	struct Params { /* Set ANN parameters. Leave default when uncertain */
		val_t _stepFactor{ 1 };

		/* Activation Function and Derivatives */
		Vector_t(*activationFunc)(const Vector_t&) { &MLMath::sigmoid };
		val_t(*dn_db)(const Vector_t&, size_t) { &MLMath::noFunc_dn_db };
		val_t(*dn_dw)(const Vector_t& n, size_t node, val_t nodeBefore) { &MLMath::noFunc_dn_dw };
		val_t(*dN_dn)(const Vector_t& n, size_t nextLayerNode, val_t weightBetween) { &MLMath::noFunc_dN_dn };
	};
	Params _params;

	explicit ANN(const std::vector<size_t>& layerSizes);
	explicit ANN(DataReader& dr); /* ANN based off DataReader with no hidden layers */
	explicit ANN(DataReader& dr, Params& params);	

	Vector_t& train();
	Vector_t& processInput();
	Vector_t& const inputs();
	Vector_t& const label();
	void readNext();
	void insertLayer(size_t size);

	/* ANN Operator Overloads */
	friend std::ostream& operator<<(std::ostream& os, const ANN& ann);

	friend const void LogicalAND::testAssertions(const ANN&); /* TODO: check position of const */
	friend const void MNISTReader::testAssertions(const ANN&);

private:
	DataReader& _dr;
	Vector_t _input;
	Vector_t _label;
	//bool _simultaneousChanges{ true };
	const std::vector<size_t> _layerSizes;
	std::vector<Vector_t> _layers;
	int _curLayer;
	std::vector<Matrix_t> _weights;
	std::vector<val_t> _biases; 

	void setInput(const Vector_t& input);
	void processLayer(size_t layer);

	/* ANN Train */
	void backprop(const Vector_t& output, const Vector_t& label);
	Vector_t backpropLayer(size_t layer, const Vector_t& backpropagated_dE_dn);
	void adjustWeights(size_t weightLayer, const Vector_t& backpropagated_dE_dn);
	void adjustBiases(size_t weightLayer, const Vector_t& backpropagated_dE_dn); 
	Vector_t backprop_dE_dn(size_t weightLayer, const Vector_t& backpropagated_dE_dn);

	/* ANN Traversal Helpers */
	val_t nodeValBefore(size_t weightLayer, size_t row, size_t col);
	val_t nodeValAfter(size_t weightLayer, size_t row, size_t col);
	size_t nodeLayerAfter(size_t weightLayer);
	size_t nodeLayerBefore(size_t weightLayer);
	val_t weight(size_t weightLayer, size_t nodeBefore, size_t nodeAfter);
	size_t weightLayerAfter(size_t nodeLayer);
	size_t lastWeightLayer();
	size_t outputLayer();

	/* ANN Calculations */
	Vector_t prepLayerAfter(size_t nodeLayer);
	Vector_t output_dE_dn(const Vector_t& output, const Vector_t& label);
	void normalize(Vector_t& vec); /* TODO: Move to MLMath, return Vector_t */
	void scale(Vector_t& vec);
};
