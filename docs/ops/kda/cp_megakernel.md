# CP backward megakernel

`Config(cp_megakernel=True)` selects the complete CP backward core from
primatrix/pallas-kernel `c82625a2bdadd5881b12f1ab37ec3fa2004c9780`.
It is a separate opt-in from the post-collective local fusion in #10/#11.
The adapter enables it only for active CP with K/V multiples of 128.
Existing public APIs, forward state policy and aligned residuals are retained.

## Single-call boundary

One Pallas call contains optional forward state reconstruction, local CP
preprocessing, synchronized remote-DMA state exchange and the reverse gradient
traversal. The ring uses an adjacent-rank shortcut for one-boundary chains and
a full ring for longer chains; a barrier precedes remote writes into scoped
VMEM. Separate ping-pong buffers serve input and output DMA.

Gate-prefix reconstruction, gradient postprocessing and scalar CP metadata
remain outside this call. The source's FP32 reconstructed h scratch is still
written to HBM inside the call in rematerialization mode. There are no separate
HBM bridge arrays for w/u/qg/kg/v_new/dAqk/dv0. Native packed DMA variants are
not enabled: the kernel consumes Tokamax's existing aligned token layout.

The incoming CP state is in Tokamax's local slot zero, independently of the
original segment ID. The rematerialization port explicitly uses that state
only for the first local segment and zeros for later segments. This fixes an
integration error exposed by a cross-rank segmented rematerialization test.

The source retains more FP32 intermediates than staged BF16 and uses
compensated BF16 intra contractions. Tests compare both staged and megakernel
gradients with the independent XLA recurrence; bitwise equality is not claimed.

## Validation

The suite runs saved/rematerialized BF16/FP32 cases, continuous and segmented
sequences, on two and four devices. It uses `pltpu.InterpretParams` with race
detection on CPU to simulate real DMA, semaphores and barriers. The ordinary
Pallas interpreter cannot execute this communication path. Invoke with
`XLA_FLAGS=--xla_force_host_platform_device_count=4`.

The existing CP benchmark compares staged, local fusion and full megakernel
forward+VJP on 2/4 TPU devices in both state policies using the device timer.
TPU lowering, VMEM feasibility, multi-host execution and performance remain
pending. Keep the option off by default and the PR draft until verified on TPU.

Local results: 16 CP cases passed under the TPU interpreter with race detection;
66 existing regression tests and 9 subtests passed; CI shard tests 33 passed;
shard consistency passed; 12 TPU benchmark cases collected without timing.
