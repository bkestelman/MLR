#pragma once
#include<vector>
#include<iostream>
#include<Eigen\Dense>
#include "Data\DataReader.h"
#include "Data\LogicalAND\LogicalAND.h"
#include "Data\MNISTReader\MNISTReader.h"
#include "MLMath.h"
#include "ANNLog.h"
#include "ANNParams.h"

class ANN {
public:
	using Matrix_t = Eigen::MatrixXd;
	using Vector_t = Eigen::VectorXd;
	using val_t = double;
	using data_t = int; /* Move to DataReader or use <> (TODO: what is this?) */

	ANNParams _params;

	ANNLog _log;

	explicit ANN(const std::vector<size_t>& layerSizes);
	explicit ANN(DataReader& dr); /* ANN based off DataReader with no hidden layers */
	explicit ANN(DataReader& dr, ANNParams& params);	

	Vector_t& train();
	void test();
	Vector_t& const inputs();
	Vector_t& const label();
	void readNext(); /* read next data and label from DataReader */
	void insertLayer(size_t size);

	void log(std::string file);
	int correct();

	/* ANN Operator Overloads */
	friend std::ostream& operator<<(std::ostream& os, const ANN& ann);

	friend const void LogicalAND::testAssertions(const ANN&); /* TODO: check position of const */
	friend const void MNISTReader::testAssertions(const ANN&);

private:
	DataReader& _dr;
	Vector_t _input; /* set by readNext() */
	Vector_t _label; /* set by readNext() */
	const std::vector<size_t> _layerSizes;
	std::vector<Vector_t> _layers;
	int _curLayer; /* TODO: delete */
	std::vector<Matrix_t> _weights;
	std::vector<val_t> _biases; 
	int _correct;
	int _tests;
	int _trains;

	void setInput(const Vector_t& input);
	Vector_t& processInput();
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
