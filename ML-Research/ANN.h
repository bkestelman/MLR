#pragma once
#include<vector>
#include<iostream>
#include<Eigen\Dense>

class ANN {
public:
	using Matrix_t = Eigen::MatrixXd;
	using Vector_t = Eigen::VectorXd;
	using val_t = double;

	explicit ANN(const std::vector<size_t>& layerSizes);

	Vector_t& train(const Vector_t& input, const Vector_t& label);
	Vector_t& processInput(const Vector_t& input);

	/* ANN Operator Overloads */
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
	static Vector_t sigmoid(const Vector_t& vec);
	static val_t sigmoid(val_t val);
	static Vector_t noFunc(const Vector_t& vec);
	Vector_t(*activationFunc)(const Vector_t&);
	Vector_t output_dE_dn(const Vector_t& output, const Vector_t& label);
};