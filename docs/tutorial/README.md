# Tutorials

Ten short pages. Each teaches **one** thing and ends with you having run
something. Work through them in order the first time — each uses what the last
one built.

Every program here is compiled and run before it ships. If one does not work,
that is a bug worth filing.

| | Page | You learn | Framed around |
| --- | --- | --- | --- |
| 1 | [Hello, GPU](01-hello-gpu.md) | open a device, dispatch a kernel, read the result | scaling an array |
| 2 | [Writing a kernel](02-writing-a-kernel.md) | the Go subset, and what `go generate` makes | brightening an image |
| 3 | [Memory](03-memory.md) | pools, buffers, views, and who closes what | weights that outlive a frame |
| 4 | [Graphs](04-graphs.md) | record once, replay many | a simulation step |
| 5 | [Cooperation](05-cooperation.md) | shared memory and barriers | a reduction |
| 6 | [Values that are not buffers](06-uniforms.md) | uniforms, and changing one without re-recording | a runtime coefficient |
| 7 | [Tensors](07-tensors.md) | shapes and operators instead of bindings | a feed-forward block |
| 8 | [Backends](08-backends.md) | picking a device, and testing without one | shipping to machines you do not own |
| 9 | [A decode step](09-a-decode-step.md) | state that survives between submissions | generating tokens, sampling on the device |
| 10 | [Quantized weights](10-quantized-weights.md) | storage width is not compute width | a model that does not fit at full width |
| 11 | [Batching sequences](11-batching-sequences.md) | a value that differs per row is a tensor | several sequences in one step |

## Which layer do you want?

accel has two, and you can ignore the one you are not using.

- **The device layer** (`accel`) deals in buffers, kernels and command graphs.
  Use it when you have your own maths to run: simulation, image and signal
  processing, anything where you write the kernel. Tutorials 1–6.
- **The tensor layer** (`accel/tensor`) deals in shapes and operators. Use it
  for inference; you never touch a bind group. Tutorials 7 and 9–11.

## What these do not cover yet

Graphics, and the Vulkan, D3D12, OpenGL and WebGPU backends. Neither is
callable. See the [status table](../../README.md#what-works-today).

Temperature and repetition penalties as a *policy object* — the primitives are
all here and tutorial 9 composes them by hand, but nothing wraps them yet.
