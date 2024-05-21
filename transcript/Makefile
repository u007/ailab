include .env
.SILENT: openai3

google:
	GOOGLE_APPLICATION_CREDENTIALS=movingfwd-a662b-1fd00f3e345e.json python google.py

openai2:
	HF_TOKEN=$(HF_TOKEN) python openai2.py

openai3:
	python openai3.py

video:
	HF_TOKEN=$(HF_TOKEN) python video.py

video2:
	HF_TOKEN=$(HF_TOKEN) python video2.py

video3:
	HF_TOKEN=$(HF_TOKEN) python video3.py
	