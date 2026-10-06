# model-router-service — container image.
#
# Multi-stage: build a wheel/venv, then run as a non-root user on a slim base.
# The image is config-free: ALL settings come from the environment at runtime
# (see .env.example). cloud.gov injects $PORT, which __main__ maps to a 0.0.0.0
# bind automatically.
FROM python:3.12-slim AS build
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

FROM python:3.12-slim AS run
# Non-root runtime user (least privilege).
RUN useradd --create-home --uid 10001 appuser
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
USER appuser
# No secrets, hosts, or ports baked in — everything is env at runtime.
# cloud.gov sets $PORT; locally, ROUTER_PORT/ROUTER_HOST apply.
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["python", "-m", "model_router_service"]
