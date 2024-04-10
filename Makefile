help:
	@echo 'Individual commands:'
	@echo ' build            - Build the environment'
	@echo ' up               - Run the chat'
	@echo ' down             - Stop the environment'
build:
	virtualenv venv -p 3.11 && source venv/bin/activate && pip install -r requirements.txt
up:
	source venv/bin/activate && python chat.py
down:
	source venv/bin/activate && deactivate
