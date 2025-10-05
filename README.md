# 2048 Game AI Agent

## Overview

This repo is designed to implement an environment for the game 2048 along with some additional code to create ai agents that can connect to this environment and interact with it.

## Setup

This is a uv managed project. To set up the environment, run:

```bash
cd 2048env
uv install
```

Then, you can run the agent itself with the main environment file (`main.py`):

```bash
uv run main.py
```

OR you can run the multiple agent example using `multi-agent.py`:

```bash
uv run multi-agent.py
```

Finally, there's also a way to test a websocket based client and server implementation using:

```bash
uv run server.py
uv run client.py
```
 