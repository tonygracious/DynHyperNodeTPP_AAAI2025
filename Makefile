libraries:
	pip install -r requirements.txt
all_libraries:
	pip install torch_scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
	pip install torch_sparse  -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
