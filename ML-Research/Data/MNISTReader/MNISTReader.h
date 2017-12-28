#pragma once
#include "Data\DataReader.h"
#include<string>
#include<fstream>

class ANN;
class MNISTReader : public DataReader {
public:
	MNISTReader();
	DataReader::Vector_t readData() override;
	DataReader::Vector_t readLabel() override;
	const size_t dataSize() override;
	const size_t labelSize() override;
	const void testAssertions(const ANN&) override;
private:
	struct MNISTFile {
		std::string filename;
		std::ifstream fstream;
		int headers, magicNumber, items, imageWidth, imageHeight;
	};
	static const int MNISTFileCount = 4;
	MNISTFile MNISTFiles[MNISTFileCount];

	void setupFilenames();
	void setupHeaderCounts();
	void openStreams();
	void readHeaders();
	void readHeaders(MNISTFile& file);	

	int imageFileToRead();
	int labelFileToRead();

	int itemsRead;
};

