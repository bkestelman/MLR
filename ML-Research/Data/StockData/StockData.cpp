#include "StockData.h"
#include <iostream>

StockData::StockData(std::string symbol) :
	BufferedDataReader(1),
	_dataSize(1),
	_dataOffset(0),
	_predictionDistance(1),
	_label(1),
	_work(_conn),
	_stock_id(idFromSymbol(symbol))
{
}

StockData::Vector_t StockData::readDataFromSource() {
	Vector_t deltas = Vector_t::Zero(_dataSize);
	Vector_t opens = Vector_t::Zero(_dataSize+2); // one extra open to get change, another extra to get label change

	pqxx::result R = _work.exec("SELECT open FROM stockdaily WHERE stock_id=2401 ORDER BY date"); /* TODO: choose stock_id or symbol */

	//std::cout << "Found " << R.size() << " stocks:\n";
	R[_dataOffset]["open"].to(opens[0]); /* TODO: readResult() function */
	for(int i = 0; i < _dataSize; i++) {
		R[_dataOffset+i+1]["open"].to(opens[i+1]);
		//deltas[i] = opens[i+1] - opens[i];
		deltas[i] = opens[i+1] > opens[i] ? 1 : 0;
	}
	R[_predictionDistance+_dataOffset+_dataSize+1]["open"].to(opens[_dataSize+1]);
	_label[0] = opens[_dataSize+1] > opens[_dataSize] ? 1 : 0;

	//std::cout << opens << "\n";
	//std::cout << "---------\n";

	_dataOffset++;

	return deltas;
}

StockData::Vector_t StockData::readLabelFromSource() {
	return _label;
}

bool StockData::test(Vector_t output) {
	assert(output.size() == 1);
	assert(_label.size() == 1);
	return abs(_label[0] - output[0]) < 0.5;
}

const size_t StockData::dataSize() {
	return _dataSize;
}
const size_t StockData::labelSize() {
	return _label.size();
}

std::string StockData::log() {
	return "StockData";
}

void StockData::useTestSet() {

}
void StockData::useTrainSet() {
	seek(0);
}
void StockData::seek(int pos) {
	_dataOffset = pos;
}

long StockData::idFromSymbol(std::string symbol) {
	pqxx::result R = _work.exec("SELECT id FROM stock WHERE symbol='" + _work.esc(symbol) + "'"); /* TODO: secure, escape symbol */ /* TODO: field names in variables */
	long ret;
	R[0][0].to(ret);
	return ret;
}	

