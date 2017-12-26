#pragma once
#include<vector>
#include<iostream>
#include<Eigen\Dense>
#include "Layer.h"

class ANN {
public:
	using Matrix_t = Eigen::MatrixXd;
	using Vector_t = Eigen::VectorXd;
	using val_t = double;

	explicit ANN(const std::vector<size_t>& layerSizes);

	Vector_t& train(const Vector_t& input, const Vector_t& label);
	Vector_t& processInput(const Vector_t& input);

	friend std::ostream& operator<<(std::ostream& os, const ANN& ann);

private:
	bool _simultaneousChanges{ true };
	double _testStep{ 0.1 };
	double _stepFactor{ 0.1 };
	const std::vector<size_t> _layerSizes;
	std::vector<Vector_t> _layers;
	int _curLayer;
	std::vector<Matrix_t> _weights;
	std::vector<val_t> _biases; 

	void setInput(const Vector_t& input);
	void processLayer(size_t layer);
	void backprop(const Vector_t& output, const Vector_t& label);
	Vector_t backpropLayer(size_t layer, const Vector_t& dE_dn_afterWeightLayer);
	double scalarError(const Vector_t& output, const Vector_t& label); // add function pointer, possibly vector error
	bool outputLayer(size_t layer);
	Vector_t dE_dn(const Vector_t& output, const Vector_t& label);

	val_t nodeBefore(size_t weightLayer, size_t row, size_t col);
	val_t nodeAfter(size_t weightLayer, size_t row, size_t col);
	size_t nodeLayerAfter(size_t weightLayer);
	size_t nodeLayerBefore(size_t weightLayer);
	val_t weight(size_t weightLayer, size_t nodeBefore, size_t nodeAfter);
	size_t lastWeightLayer();
	size_t outputLayer();
};