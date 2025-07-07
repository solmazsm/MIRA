CXX = g++
CXXFLAGS = -std=c++17 -O3

SOURCES = main.cpp coarse_fine_multivector.cpp Preprocess.cpp divGraph.cpp fastGraph.cpp basis.cpp Query.cpp
HEADERS = coarse_fine_multivector.h Preprocess.h divGraph.h fastGraph.h basis.h Query.h

TARGET = VectorSearch

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)
