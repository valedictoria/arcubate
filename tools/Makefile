# Builds the offline book tool from the engine sources. It is not shipped.
SRC := $(filter-out ../src/main.cpp,$(wildcard ../src/*.cpp ../src/nnue/*.cpp ../src/nnue/features/*.cpp))
book_tool: book_tool.cpp $(SRC)
	g++ -std=c++17 -O2 -march=sandybridge -DNDEBUG -DIS_64BIT -DUSE_POPCNT -DUSE_SSE2 -DUSE_SSSE3 -DUSE_SSE41 -I../src -Wa,-I../src -pthread book_tool.cpp $(SRC) -o book_tool
clean:
	rm -f book_tool
