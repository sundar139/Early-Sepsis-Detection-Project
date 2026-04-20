"""Serving package exposing the FastAPI application."""

from early_sepsis.serving.api import app, create_app

__all__ = ["app", "create_app"]
