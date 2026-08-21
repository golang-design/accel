# Contributing

Thanks for looking. This project is at the design stage, which is an unusually
good time to get involved: the decisions that are hardest to change later are
being made right now, and an argument against one of them is worth more than a
patch.

## What is most useful right now

**Tell us where the design is wrong.** Especially if you have shipped GPU code,
or run models in production, and something here looks naive. Open an issue.
[`docs/architecture.md`](docs/architecture.md) is the readable overview;
[`specs/`](specs/) has the full reasoning and, at the end of every spec, a list
of open questions we have not resolved.

**Backend conventions we have missed.** If you know a place where two GPU
backends disagree and it is not in [`docs/conventions.md`](docs/conventions.md),
that is directly valuable. Those entries cost hours each to discover.

**Prior art.** If another project solved one of the open questions well, saying
so saves us finding out the hard way.

Code contributions are welcome too, but until the API surface settles they carry
a real risk of being invalidated by a design change. Ask in an issue first so
neither of us wastes the effort.

## Ground rules

### No cgo

Not in the library, not in tests, not behind a build tag. Every backend reaches
its driver through [purego](https://github.com/ebitengine/purego) or raw
syscalls.

CI greps for `import "C"` rather than relying on the build, because a file can
import C behind a tag the CI platform does not select and still break someone
else. This is the one rule with no exceptions: if a feature needs cgo, it does
not go in.

### The CPU backend is the oracle

Anything a GPU backend does, the CPU backend does too, and identically. A GPU
path with no CPU equivalent has no way to be verified, so it does not merge.

### Capabilities are explicit

If a backend cannot do something, say so through the capability system and fail
with an error naming what is missing. Never silently substitute a different
result, and never silently fall back to another backend. A user whose code
quietly ran on the CPU for six months is worse off than one who got an error on
day one.

### Costs go in the docs

Every design document here states what its choice gives up, not just what it
buys. If you add a feature with a real tradeoff, write the tradeoff down. This is
a house style and reviewers will ask for it.

### Style

- `gofmt`, and CI checks it.
- Doc comments explain **why**, not just what. The why is the part a reader
  cannot reconstruct.
- No em dashes in prose. Commas, colons, and parentheses do the job.
- Commit messages explain the reasoning behind a change, not only its content.

## Getting started

```sh
git clone https://github.com/golang-design/accel
cd accel
CGO_ENABLED=0 go build ./...
CGO_ENABLED=0 go test ./...
```

Requires Go 1.27 or later. There is nothing to install and no GPU required,
which is deliberate and should stay true.

Everything currently returns `ErrNotImplemented`. That is expected: the API
surface exists so the design can be read as Go and checked by the compiler.

## How the repository is organised

| Path | What it is | Who it is for |
| --- | --- | --- |
| `docs/` | Documentation | People using or contributing to accel |
| `specs/` | Internal design specs and decisions | People building or reviewing accel |
| `*.go` | The device layer API surface | Compiles, does nothing yet |

If you change behaviour that a spec describes, update the spec in the same
change. A spec that no longer matches reality is worse than no spec.

## Reporting a problem

For a design objection, quote the specific claim you disagree with and say what
you would do instead. For a bug, once there is code to have bugs, include the
backend, the platform, and ideally a case that behaves differently on the CPU
backend than on a GPU one, since that difference is usually the whole story.

## License

Contributions are under [BSD-3-Clause](LICENSE), the same as the project.
