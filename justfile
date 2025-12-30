# Optional convenience commands (requires `just`)

setup:
	uv sync

test:
	uv run pytest

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

sample:
	uv run ml-baseline make-sample-data

train:
	uv run ml-baseline train --target is_high_value

predict:
	uv run ml-baseline predict --run latest --input data/processed/features.csv --output outputs/preds.csv
