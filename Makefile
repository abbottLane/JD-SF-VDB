# Makefile that loads a python venv

.env: requirements.txt
	@echo "Creating virtual environment..."
	@python3 -m venv .venv
	@echo "Installing requirements..."
	@.venv/bin/pip3 install -r requirements.txt
	@echo "Done."

index: .env
	@echo "Creating index..."
	@.venv/bin/python3 index.py
	@echo "Done."

search: .env
	@echo "Searching..."
	@.venv/bin/python3 search.py
	@echo "Done."
	
clean:
	@echo "Cleaning up..."
	@rm -rf .venv
	@rm -rf index
	@echo "Done."