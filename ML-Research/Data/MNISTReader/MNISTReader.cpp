#include "MNISTReader.h"
#include "SystemInfo.h"
#include "VectorCalculations.h"
#include<string>
#include<fstream>
#include<algorithm>
#include<vector>
#include<iostream>

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

MNISTReader::MNISTReader(int dataStep) : /* TODO: params struct */
	BufferedDataReader(100), /* TODO: let user choose bufferSize */
	itemsRead(0),
	_debug_log("mnist.log"),
	_dataStep(dataStep),
	_scaleDown(255),
	_labelCounts(DataReader::Vector_t::Zero(10)), /* TODO: what is this */
	imageFileToRead(TRAIN_IMAGES_FILE),
	labelFileToRead(TRAIN_LABELS_FILE)
{
	setupFilenames();
	setupHeaderCounts();
	openStreams();
	readHeaders();
}

void MNISTReader::useTestSet() {
	imageFileToRead = TEST_IMAGES_FILE;
	labelFileToRead = TEST_LABELS_FILE;
	seek(0);
}
void MNISTReader::useTrainSet() {
	imageFileToRead = TRAIN_IMAGES_FILE; /* TODO: useDataFile(file) */
	labelFileToRead = TRAIN_LABELS_FILE;
	seek(0);
}
void MNISTReader::seek(int pos) {
	MNISTFile& imageFile = MNISTFiles[imageFileToRead]; /* TODO: should imageFileToRead just be an MNISTFile? */
	std::cout << "seeking to " << imageFile.dataPos << " + " << pos*dataSize() << " * " << _dataStep << "\n";
	imageFile.fstream.seekg(imageFile.dataPos + pos*dataSize()*_dataStep); 
	std::cout << "tellg " << imageFile.fstream.tellg() << "\n";
	MNISTFile& labelFile = MNISTFiles[labelFileToRead];
	labelFile.fstream.seekg(labelFile.dataPos + pos);
	BufferedDataReader::prepareBuffer();
}
	
DataReader::Vector_t MNISTReader::readDataFromSource() {
	MNISTFile& file = MNISTFiles[imageFileToRead];
	int length = dataSize()*_dataStep;
	char* bytes = (char*)malloc(length);
	file.fstream.read(bytes, length);
	assert(!file.fstream.eof());
	assert(!file.fstream.bad());
	assert(!file.fstream.fail());
	assert(file.fstream.gcount() == length);
	DataReader::Vector_t data(length/_dataStep); /* TODO: will this work for uneven dataSteps? */
	for (int byte = 0; byte < length/_dataStep; byte++) {
		int x = (unsigned char)bytes[byte*_dataStep];
		assert(x >= 0 && x <= 255);
		data[byte] = x;
		data[byte] /= _scaleDown;
	}
	free(bytes);
	return data;
}

DataReader::Vector_t MNISTReader::readLabelFromSource() {
	char* byte = (char*)malloc(1);
	MNISTFiles[labelFileToRead].fstream.read(byte, 1);
	assert(*byte >= 0 && *byte <= 9); 
	DataReader::Vector_t label = DataReader::Vector_t::Zero(10);
	label[*byte] = 1;
	assert(label.size() == 10);
	itemsRead++; /* DEPRECATED ? */
	_labelCounts(*byte)++;
	_label = label; // NEW
	return label;
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
		//std::cout << data << std::endl;
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
	file.dataPos = file.fstream.tellg();
	//std::cout << std::endl;
}

const size_t MNISTReader::dataSize() {
	return 28*28/_dataStep; /* TODO: rely on a const instead */
}

const size_t MNISTReader::labelSize() {
	return 10;
}

bool MNISTReader::test(DataReader::Vector_t output) {
	int labelIndex = maxIndex(_label); /* TODO: fix maxIndex namespace (it's in VectorCalculations.h) */
	int outputIndex = maxIndex(output);
	if (labelIndex == outputIndex) return true;
	else return false;
}

std::string MNISTReader::log() { /* TODO: use operator<< */
	return "DataReader: MNISTReader\nData Step: " + std::to_string(_dataStep) + "\nScale Down: " + std::to_string(_scaleDown);
}

//const void MNISTReader::testAssertions(const ANN& ann) {
//	assert(ann._layerSizes.size() == 2);
//	assert(ann._layers.size() == 2);
//}
