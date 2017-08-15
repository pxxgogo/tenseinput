CXX = g++
CXXFLAGS = -std=c++11
DEPS = attitude_estimator.h
OBJ = attitude_estimator.o est_wrapper.o


libest.so: $(OBJ)
	$(CXX) -shared -Wl,-soname,$@ -o $@ $^

%.o: CXXFLAGS+=-c -fPIC
