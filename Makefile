help:
	@echo 'Individual commands:'
	@echo ' build-local      - Build the environment'
	@echo ' up-local         - Run the command line chat'
	@echo ' down-local       - Stop the environment'
	@echo ' server-local     - Start the API server'
	@echo ' lint             - Lint the code with pylint and flake8 and check imports'
	@echo ' test             - Run tests'
	@echo ' build            - Build the flask container'
	@echo ' up               - Run the flask container'
	@echo ' down             - Remove containers and network'
build-local:
	virtualenv venv -p 3.11 && source venv/bin/activate && pip install -r requirements/base.txt
up-local:
	source venv/bin/activate && python chat.py
down-local:
	source venv/bin/activate && deactivate
server-local:
	source venv/bin/activate && python api/server.py
lint:
	# Lint the python code
	pylint *
	flake8
	isort --check-only --settings-path .isort.cfg *.py
test:
	# Run python tests
	pytest -v -s tests/tests.py
linttest: lint test
build:
	docker compose -f development/docker-compose.yml up --build
up:
	docker compose -f development/docker-compose.yml up
down:
	docker compose -f development/docker-compose.yml down
