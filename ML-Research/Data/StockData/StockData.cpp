#include "StockData.h"
#include <iostream>
#include <pqxx/pqxx>

StockData::StockData(size_t dataSize) :
	BufferedDataReader(1),
	_dataSize(dataSize),
	_dataOffset(0),
	_label(1)
{
	prepareBuffer(); /* TODO: should be in BufferedDataReader? */
}

StockData::Vector_t StockData::readDataFromSource() {
	Vector_t deltas = Vector_t::Zero(_dataSize);
	Vector_t opens = Vector_t::Zero(_dataSize+2); // one extra open to get change, another extra to get label change
	int month = 30; /* predict distance TODO: put somewhere */

	pqxx::connection C;
	//std::cout << "Connected to " << C.dbname() << "\n";
	pqxx::work W(C);

	pqxx::result R = W.exec("SELECT open FROM stockdaily WHERE stock_id=2401 ORDER BY date"); /* TODO: choose stock_id or symbol */

	//std::cout << "Found " << R.size() << " stocks:\n";
	R[_dataOffset]["open"].to(opens[0]); /* TODO: readResult() function */
	for(int i = 0; i < _dataSize; i++) {
		R[_dataOffset+i+1]["open"].to(opens[i+1]);
		//deltas[i] = opens[i+1] - opens[i];
		deltas[i] = opens[i+1] > opens[i] ? 1 : 0;
	}
	R[month+_dataOffset+_dataSize+1]["open"].to(opens[_dataSize+1]);
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

void StockData::seek(int pos) {
	_dataOffset = pos;
}
