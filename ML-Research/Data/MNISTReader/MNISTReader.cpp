#include "MNISTReader.h"
#include "ANN\ANN.h"
#include "SystemInfo.h"
#include<string>
#include<fstream>
#include<algorithm>

static const std::string baseDir = "Data/MNISTReader/";
static const std::string trainImages = "train-images"; 
static const std::string testImages = "test-images";
static const std::string trainLabels = "train-labels";
static const std::string testLabels = "test-labels";
static const int trainImagesMagicNumber = 2051;
static const int testImagesMagicNumber = 2051;
static const int trainLabelsMagicNumber = 2049;
static const int testLabelsMagicNumber = 2049;
static const int trainImagesHeaders = 4;
static const int testImagesHeaders = 4;
static const int trainLabelsHeaders = 2;
static const int testLabelsHeaders = 2;
static const int headerSize = 4; /* 4 bytes */
static const int magicNumberHeader = 0;
static const int itemsHeader = 1;
static const int imageWidthHeader = 2;
static const int imageHeightHeader = 3;

static const int TRAIN_IMAGES_FILE = 0;
static const int TEST_IMAGES_FILE = 1;
static const int TRAIN_LABELS_FILE = 2;
static const int TEST_LABELS_FILE = 3;

MNISTReader::MNISTReader() :
	itemsRead(0),
	log("mnist.log")
{
	setupFilenames();
	setupHeaderCounts();
	openStreams();
	readHeaders();
}

DataReader::Vector_t MNISTReader::readData() {
	int f = imageFileToRead();
	MNISTFile& file = MNISTFiles[f];
	int length = file.imageWidth * file.imageHeight;
	assert(length == dataSize());
	char* bytes = (char*)malloc(length);
	file.fstream.read(bytes, length);
	assert(!file.fstream.eof());
	assert(!file.fstream.bad());
	assert(!file.fstream.fail());
	assert(file.fstream.gcount() == length);
	/*for (int i = 0; i < length; i++) {
		file.fstream.read(bytes, 1);
		log << "i : " << i << "\ngcount: " << file.fstream.gcount() << "\n";
		if (file.fstream.fail()) log << "fail\n";
		if (file.fstream.eof()) log << "eof\n";
		if (file.fstream.bad()) log << "bad\n";
		//char* error = (char*)malloc(strerrorlen_s(errno));
		//log << "Error: " << strerror(errno) << "\n";
	}*/
	//assert(MNISTFiles[f].fstream.gcount() == length);
	DataReader::Vector_t data(length);
	for (int byte = 0; byte < length; byte++) {
		int x = (unsigned char)bytes[byte];
		log << x << "\n";
		assert(x >= 0 && x <= 255);
		data[byte] = x;
	}

	log << data;
	free(bytes);
	return data;
}

DataReader::Vector_t MNISTReader::readLabel() {
	int f = labelFileToRead();
	char* byte = (char*)malloc(1);
	MNISTFiles[f].fstream.read(byte, 1);
	assert(*byte >= 0 && *byte <= 9); 
	ANN::Vector_t label = ANN::Vector_t::Zero(10);
	//std::cout << label << std::endl << std::endl;
	label[*byte] = 1;
	assert(label.size() == 10);
	//std::cout << label;
	itemsRead++;
	return label;
}

int MNISTReader::imageFileToRead() {
	if (itemsRead < MNISTFiles[TRAIN_IMAGES_FILE].items) return TRAIN_IMAGES_FILE;
	else return TEST_IMAGES_FILE;
}
int MNISTReader::labelFileToRead() {
	if (itemsRead < MNISTFiles[TRAIN_LABELS_FILE].items) return TRAIN_LABELS_FILE;
	else return TEST_LABELS_FILE;
}

void MNISTReader::setupFilenames() {
	MNISTFiles[TRAIN_IMAGES_FILE].filename = trainImages;
	MNISTFiles[TEST_IMAGES_FILE].filename = testImages;
	MNISTFiles[TRAIN_LABELS_FILE].filename = trainLabels;
	MNISTFiles[TEST_LABELS_FILE].filename = testLabels;
}

void MNISTReader::setupHeaderCounts() {
	MNISTFiles[0].headers = trainImagesHeaders;
	MNISTFiles[1].headers = testImagesHeaders;
	MNISTFiles[2].headers = trainLabelsHeaders;
	MNISTFiles[3].headers = testLabelsHeaders;
}

void MNISTReader::openStreams() {
	for (int f = 0; f < MNISTFileCount; f++) {
		MNISTFiles[f].fstream = std::ifstream(baseDir + MNISTFiles[f].filename, std::ios::binary);
		assert(MNISTFiles[f].fstream.is_open());
	}
}

void MNISTReader::readHeaders() {
	assert(MNISTFileCount > 0);
	for (int f = 0; f < MNISTFileCount; f++) {
		readHeaders(MNISTFiles[f]);
	}
}

void MNISTReader::readHeaders(MNISTFile& file) {
	assert(file.headers > 0);
	assert(headerSize == 4); /* MNIST file header size */
	char data[headerSize];
	for (int header = 0; header < file.headers; header++) {
		file.fstream.read(data, headerSize);
		assert(file.fstream.is_open());
		std::cout << data << std::endl;
		if (!SystemInfo::bigEndian) std::reverse(data, data + 4);
		int* ip = (int*)&data;
		switch (header) {
		case magicNumberHeader:
			file.magicNumber = *ip;
			assert(*ip == 2051 || *ip == 2049); /* Possible magic number for MNIST files */
			break;
		case itemsHeader:
			file.items = *ip;
			assert(*ip == 10000 || *ip == 60000);
			break;
		case imageWidthHeader:
			file.imageWidth = *ip;
			assert(*ip == 28);
			break;
		case imageHeightHeader:
			file.imageHeight = *ip;
			assert(*ip == 28);
			break;
		default:
			assert(header < file.headers && header > 0); 
		}
	}
	std::cout << std::endl;
}

const size_t MNISTReader::dataSize() {
	return 28*28; /* TODO: rely on a const instead */
}

const size_t MNISTReader::labelSize() {
	return 10;
}

const void MNISTReader::testAssertions(const ANN& ann) {
	assert(ann._layerSizes.size() == 2);
	assert(ann._layers.size() == 2);
}