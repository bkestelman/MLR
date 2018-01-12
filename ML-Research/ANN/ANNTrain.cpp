#include "ANN.h"
#include<numeric>

void ANN::train() {
	_trains++;
	readBatch();
	//_trainLog << "input:\n" << _inputBatch[0] << "\nlabel:\n" << _labelBatch[0] << "\noutput:\n" << processInput(0) << "\n";
	val_t errBefore = batchError();
	//_trainLog << *this << "\n";
	backprop(); 
	//_trainLog << "output after train:\n" << processInput(0) << "\n";
	//_trainLog << *this << "\n";
	val_t errAfter = batchError();
	_trainLog << "err before: " << errBefore << "\nerr after: " << errAfter << "\n";
	//assert(errBefore > errAfter);
	//for (int i = 0; i < _layers.size()-1; i++) {
	//	assert(_layers[i][_layers[i].size() - 1] = -1); /* check that biases are unchanged */
	//}
}

void ANN::backprop() {
	//_trainLog << *this << "\n";
	for (int epoch = 0; epoch < _params._iterations; epoch++) { 
		Vector_t bp_delta(_layerSizes[outputLayer()]);
		Vector_t output_delta = Vector_t::Zero(_layerSizes[outputLayer()]);
		for (int input = 0; input < _params._batchSize; input++) { /* for each input i */
			output_delta = processInput(input) - _labelBatch[input]; /* (Y(2) - d)_j */
		}
		for (int input = 0; input < _params._batchSize; input++) { /* for each input i */
			Vector_t I = weightedSum(outputLayer() - 1); /* I(2)_j */
			for (int j = 0; j < _layerSizes[outputLayer()]; j++) { /* j indexes outputs (exclude bias node) */
				bp_delta[j] = output_delta[j] * _params.activationFunc_d(I(j)); /* (d - Y(2))_j g'(I(2)_j)*/
				assert(_params.activationFunc_d(_layers[outputLayer()](j)) > 0);
			}
		}
		for (auto weightLayer = lastWeightLayer(); weightLayer > 0; weightLayer--) { /* nasty unsigned trick here */
			adjustWeights(weightLayer, bp_delta);
			bp_delta = backpropLayer(weightLayer, bp_delta);
		}
		adjustWeights(0, bp_delta);
		//commitAdjustments();
	}
}

/* dE_dn is the backpropagated derivative of error function with respect to nodes in the node layer after weightLayer */
ANN::Vector_t ANN::backpropLayer(size_t weightLayer, Vector_t& bp_delta) { /* TODO: check for memory leak */	
	//commitAdjustments(weightLayer);
	Vector_t new_delta = backprop_delta(weightLayer, bp_delta);	/* TODO: test before, test after */
	return new_delta; /* pass this to the next backPropLayer call */
}

void ANN::adjustWeights(size_t weightLayer, Vector_t& bp_delta) {
	assert(_weights[weightLayer].rows() == _layerSizes[nodeLayerAfter(weightLayer)]);
	assert(_weights[weightLayer].cols() == _layers[nodeLayerBefore(weightLayer)].size());
	for (int i = 0; i < _layers[nodeLayerBefore(weightLayer)].size(); i++) { 
		for (int j = 0; j < _layerSizes[nodeLayerAfter(weightLayer)]; j++) { 
			/* W(2)ij = W(2)ij + eta delta(2)_j Y(1)_i */
			_weights[weightLayer](j, i) -= _params._learningRate * bp_delta(j) * _layers[nodeLayerBefore(weightLayer)](i);
		}
	}

	//Vector_t& n = _layers[nodeLayerAfter(weightLayer)];
	//for (int i = 0; i < _params._batchSize; i++) {
	//	processInput(i); /* TODO: only need to process input up to nodeLayerAfter(weightLayer)... OR store processed results for each input in batch */
	//	for (auto row = 0; row < _weights[weightLayer].rows(); row++) {
	//		for (auto col = 0; col < _weights[weightLayer].cols(); col++) {
	//			val_t dn_dw = _params.dn_dw(n, row, nodeValBefore(weightLayer, row, col));
	//			val_t dE_dw = dn_dw * backpropagated_dE_dn[i][row]; /* chain rule */
	//			_weightsDeltas[weightLayer](row, col) -= dE_dw * _params._learningRate;// / abs(rowDelta);
	//		}
	//	}
	//}
}

ANN::Vector_t ANN::backprop_delta(size_t weightLayer, Vector_t& bp_delta) { 
	size_t beforeWeights = nodeLayerBefore(weightLayer); 
	size_t afterWeights = beforeWeights + 1; 
	Vector_t new_delta = Vector_t::Zero(_layers[beforeWeights].size()); 
	for (int input = 0; input < _params._batchSize; input++) {
		Vector_t I = weightedSum(nodeLayerBefore(weightLayer) - 1);
		for (int j = 0; j < _layerSizes[beforeWeights]; j++) { /* bias node also gets a bp_delta */
			for (int k = 0; k < _layerSizes[afterWeights]; k++) { /* but no bp_delta obtained from bias node (no weights to bias node) */
				new_delta[j] += bp_delta[k] * _weights[weightLayer](k, j) * _params.activationFunc_d(I[j]);
				/* delta(1)_i = -sum(delta(2)_j * W(2)_ij) g'(I(1)_i) */
			}
		}

		//for (auto node = 0; node < _layers[nodeLayerBeforeWeights].size(); node++) { 
		//	dE_dn = 0;
		//	for (auto nextLayerNode = 0; nextLayerNode < _layers[nodeLayerAfterWeights].size(); nextLayerNode++) { 
		//		val_t dN_dn = _params.dN_dn(n, nextLayerNode, weight(weightLayer, node, nextLayerNode)); // check
		//		dE_dn += dN_dn * backpropagated_dE_dn[i][nextLayerNode]; /* chain rule */ // [0]
		//	}
		//	new_dE_dn[i][node] = dE_dn;
		//}
	}
	return new_delta;
}

void ANN::commitAdjustments() { /* TODO: commit each layer as you backprop? */
	for (int i = 0; i < _weights.size(); i++) {
		_weights[i] += _weightsDeltas[i] / _params._batchSize; // NEW 
		_weightsDeltas[i] = Matrix_t::Zero(_weightsDeltas[i].rows(), _weightsDeltas[i].cols());
	}
}
void ANN::commitAdjustments(size_t weightLayer) { /* TODO: commit each layer as you backprop? */
	_weights[weightLayer] += _weightsDeltas[weightLayer] / _params._batchSize; // NEW 
	_weightsDeltas[weightLayer] = Matrix_t::Zero(_weightsDeltas[weightLayer].rows(), _weightsDeltas[weightLayer].cols());
}