run:
	make clean
	python3 gridsearch.py

logs:
	find -name 'logs.txt' -exec tail -n 50 {} +

clean:
	rm -rf epoch_cache/*

kill:
	jbadmin -kill -proj understanding_amazon -state r all

status:
	jbinfo -summary -proj understanding_amazon -state r
