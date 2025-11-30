all:
	python setup.py build_ext --inplace
	@echo "Skipping setup_torch.py (GCKN C++ extensions not needed for basic GraphiT)"
