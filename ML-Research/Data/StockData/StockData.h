#pragma once
#include "Data/BufferedDataReader.h"
#include<pqxx/pqxx>
#include<string>

class StockData : public BufferedDataReader {
public:
	using Vector_t = Eigen::VectorXd;

	size_t _dataSize, _predictionDistance;

	StockData(std::string symbol);
	const size_t dataSize() override;
	const size_t labelSize() override;
	void useTestSet() override;
	void useTrainSet() override;
	void seek(int);
	bool test(Vector_t) override;
	std::string log() override;

	long idFromSymbol(std::string symbol);
private:
	pqxx::connection _conn;
	pqxx::work _work;

	long _stock_id;
	size_t _dataOffset;

	Vector_t _label;

	Vector_t readDataFromSource() override;
	Vector_t readLabelFromSource() override;

};
