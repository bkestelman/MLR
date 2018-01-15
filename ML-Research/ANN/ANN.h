#pragma once
#include<vector>
#include<iostream>
#include<Eigen/Dense>
#include "Data/DataReader.h"
#include "Data/LogicalFunctions/LogicalAND.h"
#include "Data/MNISTReader/MNISTReader.h"
#include "MLMath.h"
#include "ANNLog.h"
#include "ANNParams.h"

class ANN {
public:
	using Matrix_t = Eigen::MatrixXd;
	using Vector_t = Eigen::VectorXd;
	using val_t = double;

	ANNParams _params;

	ANNLog _log;

	explicit ANN(const std::vector<size_t>& layerSizes);
	//explicit ANN(DataReader& dr); /* ANN based off DataReader with no hidden layers */
	explicit ANN(DataReader& dr, ANNParams& params);	

	void init(); /* run necessary initialization that can't be done in constructor (ex: init weights, after layers inserted, after constructor) */
	void train();
	Vector_t& test();
	Vector_t& inputs();
	Vector_t& label();
	Vector_t& output() { return _layers[_layers.size() - 1]; };
	void readNext(); /* read next data and label from DataReader TODO: private */
	void insertLayer(size_t layer, size_t size);
	Vector_t layer(size_t l) { return _layers[l]; };

	void log(std::string file);
	int correct();

	/* ANN Operator Overloads */
	friend std::ostream& operator<<(std::ostream& os, const ANN& ann);

	//friend const void LogicalAND::testAssertions(const ANN&); /* TODO: check position of const */
	//friend const void MNISTReader::testAssertions(const ANN&);

private:
	DataReader& _dr;
	Vector_t _input; /* set by readNext() */
	Vector_t _label; /* set by readNext() */
	std::vector<size_t>& _layerSizes;
	std::vector<Vector_t> _layers;
public:
	std::vector<Matrix_t> _weights;
	std::vector<Matrix_t> _weightsDeltas;
	//std::vector<Vector_t> _biases; /* TODO: make private */
	//std::vector<Vector_t> _biasesDeltas; /* TODO: make private */
private:
	int _correct;
	int _tests;
	int _trains;
	std::vector<val_t> errors;

	void setInput();
	Vector_t& processInput();
	void processLayer(size_t layer);

	void initLayers();
	void initWeights();
	Matrix_t initWeightMatrix(size_t weightLayer);

	/* ANN Train */
	void backprop(); /* backprop batch */
	Vector_t backpropLayer(size_t layer, Vector_t& backprop_delta);
	void adjustWeights(size_t weightLayer, Vector_t& backprop_delta);
	Vector_t backprop_delta(size_t weightLayer, Vector_t& delta);
	void commitAdjustments();
	void commitAdjustments(size_t weightLayer);

	/* ANN Traversal Helpers */
	val_t nodeValBefore(size_t weightLayer, size_t row, size_t col);
	val_t nodeValAfter(size_t weightLayer, size_t row, size_t col);
	size_t nodeLayerAfter(size_t weightLayer);
	size_t nodeLayerBefore(size_t weightLayer);
	val_t weight(size_t weightLayer, size_t nodeBefore, size_t nodeAfter);
	val_t& weightDelta(size_t weightLayer, size_t nodeBefore, size_t nodeAfter);
	size_t weightLayerAfter(size_t nodeLayer) const;
	size_t weightLayerBefore(size_t nodeLayer);
	size_t lastWeightLayer();
	size_t outputLayer();
	void insertWeightsBefore(size_t layer);
	void setWeightsAfter(size_t layer);
	void setupLayer(size_t layer); /* Insert layer using value in _layerSizes for size (private, used by constructor) */
	size_t sizeWithBias(size_t nodeLayer);
	size_t sizeNoBias(size_t nodeLayer);

	/* ANN Calculations */
	Vector_t weightedSum(size_t nodeLayer) const;
	Vector_t output_dE_dn(const Vector_t& output, const Vector_t& label);
	val_t error();
	void normalize(Vector_t& vec); /* TODO: Move to MLMath, return Vector_t */
	void scale(Vector_t& vec);

	std::ofstream _trainLog{ "out.log" };

};
