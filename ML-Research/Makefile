CXXFLAGS = -I eigen/ -I . -g
OBJDIR = build
DATA = Data
ANN = ANN
OBJ = $(addprefix $(OBJDIR)/, ML-Research.o MLMath.o VectorCalculations.o SystemInfo.o \
      $(DATA)/MNISTReader/MNISTReader.o \
      $(ANN)/ANN.o $(ANN)/ANNTrain.o $(ANN)/ANNLog.o $(ANN)/ANNCalculations.o $(ANN)/ANNTraversalHelpers.o)

MLR: $(OBJ) 
	$(CXX) $(CXXFLAGS) $(OBJ) -o MLR
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ): | $(OBJDIR)
$(OBJDIR):
	mkdir $(OBJDIR)
	mkdir $(OBJDIR)/$(DATA)
	mkdir $(OBJDIR)/$(DATA)/MNISTReader
	mkdir $(OBJDIR)/$(ANN)

clean:
	rm -r build
