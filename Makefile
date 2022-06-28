make clean:
		black -l 100 shgp/
		isort --atomic -l 100 --trailing-comma --remove-redundant-aliases --multi-line 3 shgp/

make notebooks:
		jupytext --to notebook examples/*.py

make tests:
		pytest -n auto -v tests/

make environment_update:
		conda env export > environment.yml