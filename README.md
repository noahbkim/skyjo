# Skyjo

A Skyjo simulation and various attempts at optimizing gameplay. You can run the
simulation via the command line. It requires a recent Python 3, probably 3.10
to be safe.

## Quickstart

To get started, try using the command line:

```shell
# Simulate a game of Skyjo with 4 `RandomPlayer`'s. You can see the code for
# `RandomPlayer` in main.py
skyjo $ python3 main.py RandomPlayer RandomPlayer RandomPlayer RandomPlayer

# Enable interactive mode to see what the bots are doing in each turn. Use the
# CLI shorthand +3 to get four unique `RandomPlayer` instances.
skyjo $ python3 main.py -i RandomPlayer+3

# Simulate 100 games and aggregate the results.
skyjo $ python3 main.py -n 100 RandomPlayer+3

# Simulate 10000 games and use multiprocessing to parallelize work. By default,
# this will use as many cores as are available on your computer.
skyjo $ python3 main.py -m -n 10000 RandomPlayer+3

# Throw a rudimentary heuristic in the mix. Note that there is not yet support
# for automatically permuting turn order, which will potentially skew results
# from player rosters that aren't rotationally symmetric.
skyjo $ python3 main.py -m -n 10000 ThresholdPlayer RandomPlayer+2
```

Next, try implementing your own `Player` subclass by editing `main.py` or
creating a new Python module (e.g. `mymodule.py`) and instantiating it using
its qualified name, e.g.

```shell
skyjo $ python3 main.py -m -n 10000 mymodule.MyPlayer RandomPlayer+2

# You can provide arbitrary data to your `Player` subclass from the command
# line using the `:` suffix. The substring after the colon will be passed as
# the first argument to the `MyPlayer` constructor.
skyjo $ python3 main.py -m -n 10000 mymodule.MyPlayer:risky RandomPlayer+2
```
