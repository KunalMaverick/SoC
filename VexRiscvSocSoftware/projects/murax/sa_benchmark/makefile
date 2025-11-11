MARCH=rv32i_zicsr
PROJ_NAME=sa_benchmark
DEBUG=yes
BENCH=no
MULDIV=no


SRCS = 	$(wildcard src/*.c) \
		$(wildcard src/*.cpp)  \
		$(wildcard src/*.S)


INC += -I../../../libs/
INC += -I../libs/
LDSCRIPT = ../libs/linker.ld

include ../../../resources/gcc.mk
include ../../../resources/subproject.mk


			
