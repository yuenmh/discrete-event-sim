# Prototype discrete event sim

Dependencies are in the `pyproject.toml`, you can install normally or with uv.

## Examples

Most of the stuff for creating protocols comes from `des.sim`.
For creating a process that is like a server, use `SMBuilder` which is similar to the FastAPI builder
in that it has a decorator you place on a function to handle messages sent to the process.

For example, a simple process that accepts a ping message can be created like this:

```python
from des.sim import SMBuilder, Atom, Addr, Ref, send

class Ping(Atom): ...

ping_server = SMBuilder()

@ping_server.handle(Ping)
async def handle_ping(sender: Addr, ref: Ref):
    send(sender, ref, "pong")
```

You can basically think of each thing annotated with `handle` as being like an endpoint.

> **Note**: The `Atom` base class just implements equality on the class itself with instances of the class so you can send the class
> and use the `match` statement to handle it.
>
> `Addr` represents a process address, and `Ref` is a unique reference for an action. The ref thing is a common pattern from
> erlang where you send a message with a ref and the receiver responds with the same ref so you can match responses to requests.
>
> Items in the `handle` decorator are matched against the beginning of the message and stripped off before the rest of the message
> params are passed to the handler function. So in the above example, when that process receives a message like `(Ping, Addr(...), Ref(...))`,
> the `Ping` item is matched and stripped off, and the remaining items `Addr(...)` and `Ref(...)` are passed as parameters to the handler.

For simple cases when you just want to spawn a process that acts more like a client than a server, you can use the `launch` decorator.
This preloads the inbox with a message to start the process when you spawn it.

```python
# imports ...
ping_server_addr = ...

@launch
async def client_process():
    # demo of sending a ping and waiting for a response
    ref = Ref()
    send(ping_server_addr, Ping, self(), ref)
    msg = await wait(ref)

    # this is a common pattern, so there is also ask
    msg = await ask(ping_server_addr, Ping)
```

You can loop inside but there should probably be at least one `await` so it doesn't run forever.

To run the simulation use the `EventLoop` class and spawn each process with `EventLoop.spawn(addr, sm)`.
Then run with `EventLoop.run(num_epochs)` (you can decide like 1 epoch = 1ms).

## Working api for now

Functions for sending and waiting are as follows:

- `send(addr, *msg)`: send a message to the given address
- `await ask(addr, method, *args)`: constructs a message like `(method, self(), Ref(), *args)` and sends it to the address, then
  waits for a response with the same ref, and returns the first item of the response message after the ref.
- `await wait(*msg)`: waits for a message matching the given items and returns the rest of the message items.
- `await {ask,wait}_timeout(timeout, ...)`: same as above but raises `TimeoutError` if the timeout is reached.

These functions get the state from a contextvar that the event loop manages. Others are:

- `self()` or `this()`: probably a mistake originally naming it self but it gets the current process address.
- `rng()`, `now()`, `stop()`, `sleep()`, `sleep_until()`: self explanatory
- `log(msg, **kwargs)` is currently the main way of calculating statistics. The pattern I have been using is to pick
  a unique name for each type of event and then filter the log after and put the values into a dataframe to analyze.

I would consider the others to be internal.

There is a premade queue in `des.stdlib`. The `Queue` class is a wrapper around the address that sends and waits in the right way.

## Problems

Big problem: there is no way of saving the state and resuming it later. This is because the state of all processes is stored
in the call stack of coroutines, which as far as I know cannot be cloned or serialized. I am not entirely sure how to fix this.

No way to spawn from inside a process. This might be good depending on how you look at it. A workaround is just spawn everything
that is needed, and wake up at the predetermined time.

I want to figure out a way to make the at least the send typesafe. Current ideas are to require properties with type annotations
on the `Atom` subclasses, then check at runtime that all the required args are there; or maybe something can be done with `typing.ParamSpec`.
