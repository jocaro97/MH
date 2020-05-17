# Directorios del proyecto
SRC = FUENTES
BIN = BIN

# Opciones de compilación
CXX = g++
CXXFLAGS = -std=c++11 -g -I./$(INC) -O3

# Archivos del proyecto
SOURCES3 := p3.cpp
SOURCES3 := $(addprefix $(SRC)/, $(SOURCES3))
SOURCES2 := p2.cpp
SOURCES2 := $(addprefix $(SRC)/, $(SOURCES2))
SOURCES1 := p1.cpp
SOURCES1 := $(addprefix $(SRC)/, $(SOURCES1))

.PHONY: clean mrproper

# ************ Generación de ejecutables *************

all: p1 p2 p3
p1: $(BIN)/p1
p2: $(BIN)/p2
p3: $(BIN)/p3

# -- Práctica 3 --
$(BIN)/p3:
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES3)

# -- Práctica 2 --
$(BIN)/p2:
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES2)

# --- Práctica 1 ---
$(BIN)/p1:
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES1)

# ************ Limpieza ************
clean :
	-@rm -f $(OBJ)/* $(SRC)/*~ 
	@echo "Limpiando..."

mrproper : clean
	-@rm -f $(BIN)/*
	@echo "Limpieza completada"
