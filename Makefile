
run:
	make clean
	python3 gridsearch.py

logs:
	find -name 'logs.txt' -exec tail -n 50 {} +

clean:
	rm -rf epoch_cache/*
