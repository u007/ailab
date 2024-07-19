include .env
.SILENT: openai3

google:
	GOOGLE_APPLICATION_CREDENTIALS=movingfwd-a662b-1fd00f3e345e.json python google.py

openai2:
	python openai2.py

openai3:
	python openai3.py

openai5:
	python openai5.py

openai6:
	python openai6.py

openai7:
	python openai7.py

video:
	HF_TOKEN=$(HF_TOKEN) python video.py

video2:
	HF_TOKEN=$(HF_TOKEN) python video2.py

video3:
	HF_TOKEN=$(HF_TOKEN) python video3.py
	