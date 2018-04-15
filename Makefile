TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

LFLAGS = $(TF_LFLAGS)
CC = g++
CFLAGS = -std=c++11 -shared -Wall $(TF_CFLAGS) -O2 -fPIC

OBJECTS = $(patsubst %.cc, %.so, $(wildcard *.cc))
TARGET = prog

.PHONY: default all clean

default: $(TARGET)
all: default

%.so: %.cc
	$(CC) $(CFLAGS) $< -o $@ $(LFLAGS)

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)

clean:
	-rm -f *.so
	-rm -f $(TARGET)
