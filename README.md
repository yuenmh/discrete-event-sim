# Prototype discrete event sim

## Installation

The dependencies are in `pyproject.toml`, so it can be installed with `pip install -e .` or with `uv`. I would say `uv` is much easier
because it automatically creates the venv and makes it very easy to download python versions.

Linters are specified in the makefile.

## Basic concepts

The core primitives are processes that communicate by sending messages to each other.
Messages are basically tuples of arbitrary objects, and state machines expose patterns that specify what shape of messages they
can handle in their current state. Each process has an inbox where
messages are stored until its associated state machine is able to handle them. Processes can also spawn other state machines
to create new processes.

The main entrypoint of the library is the `EventLoop` class, which implements the message passing and process scheduling.
The workflow for running simulations is to define the state machines, add the root processes to the event loop to be spawned
at startup, then run the event loop.

### Important types

- `Addr`: address of a process. This is what is passed into `send` and similar functions to identify the target process.
- `Ref`: unique reference for matching responses to requests.

The canonical call with no response pattern is to send a message of the form `(method, *args)` to the target address.
If the result is needed, then the pattern is `(method, sender_addr, ref, *args)`. Then the receiver can send the result
back to `sender_addr` with the pattern `(ref, result)`.

## Defining state machines

To define a state machine, subclass `StateMachineBase`. Methods decorated with `@handle()` become message handlers that are
called by sending messages to the process address. This base class creates two things: the state machine implementation;
and a proxy object that hides the low-level `send` calls.

The following is an example of a simple counter state machine:

```python
class Counter(StateMachineBase):
    def __init__(self, start: int = 0):
        self.counter = start

    @handle()
    async def inc(self):
        self.counter += 1

    @handle()
    async def _get(self, sender: Addr, ref: Ref):
        send(sender, ref, self.counter)

    async def get(self) -> int:
        """wrapper around get for the proxy object"""
        return await ask(addr_of(self), Counter._get)
```

Each async function decorated with `handle` implements how to handle a specific message. If `handle` is called with no arguments as shown,
the state machine reacts to messages that start with the decorated function object. Otherwise, the arguments specify the pattern to match.
These functions become non-async methods on the proxy object that send messages to the process address.

Methods not decorated with `handle` are just normal functions and are present in both the state machine and the proxy object. This means
that you must decide whether a method is intended to be used by a message handler (in the process) or as part of the proxy object (by other processes).

The normal constructor creates the state machine implementation. To create the proxy object, use `interface` to wrap an existing address, or
`spawn_interface` to spawn a new process and wrap its address with a proxy.

From the implementation of another state machine, the counter can be spawned and used as follows:

```python
counter = spawn_interface(Counter())
counter.inc()
value = await counter.get()
```

> [!NOTE]
> I am planning to add a way of automatically creating a "getter" type method like `get` in the above example. The challenge is finding something
> that still allows type checking.

To define a state machine that acts more like a main function, or as a client as opposed to a server, subclass `LaunchedStateMachine` and override `start`,
or use the `@launch` decorator on an async function. This decorator turns the function into a state machine.

## API for implementing state machines

Functions for sending and waiting are as follows:

- `send(addr, *msg)`: send a message to the given address
- `await ask(addr, method, *args)`: constructs a message like `(method, this(), Ref(), *args)` and sends it to the address, then
  waits for a response with the same ref, and returns the first item of the response message after the ref.
- `await wait(*msg)`: waits for a message matching the given items and returns the rest of the message items.
- `await {ask,wait}_timeout(timeout, ...)`: same as above but raises `TimeoutError` if the timeout is reached.
- `spawn(state_machine)`, `spawn_interface(state_machine)`: spawn a new process running the given state machine.
  The former returns the new address, the latter returns a proxy object.

These functions get the state from a contextvar that the event loop manages. Others are:

- `this()`: Gets the current process address.
- `rng()`, `now()`, `stop()`, `sleep()`, `sleep_until()`: self explanatory
- `log(msg, **kwargs)` is currently the main way of calculating statistics. The pattern I have been using is to pick
  a unique name for each type of event and then filter the log after running the sim and put the values into a dataframe to analyze.

## Premade state machines

There are premade fundamental state machines in `des.stdlib`.

- `Queue`: FIFO queue. I need to rewrite it to use the new class-based state machine system.
- `WaitGroup`: similar to `sync.WaitGroup` in go. The purpose is to join multiple child processes.

`LaunchedStateMachine` is a base class for a state machine that runs immediately on spawn.
