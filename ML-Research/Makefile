CXXFLAGS = -I eigen/ -I . -g -lpqxx -lpq
OBJDIR = build
DATA = Data
ANN = ANN
OBJ = $(addprefix $(OBJDIR)/, ML-Research.o MLMath.o VectorCalculations.o SystemInfo.o \
      $(DATA)/BufferedDataReader.o \
      $(DATA)/LogicalFunctions/LogicalXOR.o \
      $(DATA)/MNISTReader/MNISTReader.o \
      $(DATA)/StockData/StockData.o \
      $(ANN)/ANN.o $(ANN)/ANNTrain.o $(ANN)/ANNLog.o $(ANN)/ANNCalculations.o $(ANN)/ANNTraversalHelpers.o \
      StockAlgorithms.o )

MLR: $(OBJ) 
	$(CXX) $(OBJ) $(CXXFLAGS)
$(OBJDIR)/%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJ): | $(OBJDIR)
$(OBJDIR):
	mkdir $(OBJDIR)
	mkdir $(OBJDIR)/$(DATA)
	mkdir $(OBJDIR)/$(DATA)/MNISTReader
	mkdir $(OBJDIR)/$(DATA)/LogicalFunctions
	mkdir $(OBJDIR)/$(DATA)/StockData
	mkdir $(OBJDIR)/$(ANN)

clean:
	-rm -r build
	-rm a.out
	-rm MLR
