# Skyjo

AI model, training, and gameplay for Skyjo

## Usage

Install the project in editable mode while developing:

```sh
uv sync
```

Then import the core game API directly from `skyjo`, or import supporting
modules from the package:

```python
import skyjo as sj
from skyjo import play, skynet

state = sj.new(players=2)
model_cls = skynet.EquivariantSkyNet
```
