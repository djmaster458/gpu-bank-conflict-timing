OBJS=sbox_encrypt.ptx

# No optimization, higher compute versions may do multi-cast warp optimization -> untested 
NVCC=nvcc
NVCCFLAGS= -O0

.PHONY: all
all: $(OBJS)

.PHONY: clean
clean:
	$(RM) $(OBJS)

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -ptx $^ -o $@
