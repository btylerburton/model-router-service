# cloud.gov / Cloud Foundry Python buildpack process definition.
#
# The buildpack installs requirements.txt (deps only) and runs this command. Our
# package lives under src/ (src-layout) and is NOT pip-installed by the buildpack,
# so we add src/ to PYTHONPATH here. __main__ reads $PORT and binds 0.0.0.0:$PORT.
web: PYTHONPATH=src python -m model_router_service
