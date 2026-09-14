# Tensor Mesh

Tensor Mesh trains a set of two dimensional embeddings to reproduce a handwritten word. The model runs in PyTorch on the server. The browser turns its coordinates into a physical mesh and lets the pointer alter the weights while training continues.

[kivilcimlab.org/tensormesh](https://kivilcimlab.org/tensormesh)
## Building the target

The browser rasterises `alsopasha` on an offscreen canvas and samples pixels whose brightness exceeds 128. Sampling begins at 7 pixel intervals and tightens when the viewport calls for more nodes. The target node budget scales with screen area and is bounded between 400 and 3500.

Each sampled point becomes the target for one row of `nn.Embedding(N, 2)`. There is no hidden layer. The two values in a row are the predicted x and y coordinates themselves.

The embedding starts around the centre of the unit square. SGD with a learning rate of `0.01` minimises squared error against the target coordinates. Roughly every 66 ms, the server sends the mean loss, gradient norm, and all predicted positions over the WebSocket.

## Turning training into motion

Model predictions are targets for the renderer, not the positions drawn directly. Every visible node has velocity. A force pulls it towards the latest prediction, neighbouring nodes are joined by springs based on proximity in the finished word, and velocity damping removes energy from the system.

This extra physical layer makes optimisation readable as motion rather than a sequence of teleports. Before the backend connects, the same nodes remain in a compact random cloud.

## Editing the model

The pointer repels nearby nodes in the mesh. Their displaced coordinates are normalised and sent to the backend, where valid values are written directly into `coords.weight` under `torch.no_grad()`. The disturbance is therefore a real parameter edit. Loss rises and the existing optimiser pulls the affected rows back towards their targets.

The WebSocket endpoint limits connection count, message size, node count, origins, coordinate range, and embedding indices before accepting client data.

## Structure

`backend/` contains the FastAPI application and PyTorch training loop.

`frontend/` contains the React interface, Vite configuration, target sampling, mesh physics, and Canvas 2D renderer. Set `VITE_BACKEND_URL` to the HTTP origin of the backend; the client derives the WebSocket URL from it.
