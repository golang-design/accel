---
name: land
description: After pushing code to accel, bring everything else back in step - specs, README, docs, package docs, and CI. Audits what shipped against what is written down, records outcomes and deviations in the owning spec, and watches CI to green. Use after landing a change, finishing a milestone slice, or when asked to "land", "wrap up", or "check CI".
---

# Land a change

Pushing code is the middle of the job. This is the rest of it: make every
document that describes the code true again, and make CI green.

Run it after a push. It is cheap when nothing drifted and it is the only thing
that catches the case where something did.

## Order

Do these in order. CI first, because a red build makes everything else
premature, and the audit last, because it depends on knowing what shipped.

### 1. CI

```sh
gh run list --limit 5
```

If the latest run failed:

```sh
gh run view <id>                 # which job
gh run view <id> --log-failed    # why
```

**Check whether earlier runs share the cause** before fixing. One bug produces
a run of red commits, and fixing what the newest log shows without looking back
means fixing a symptom.

Then, before anything else, decide **which of the two is wrong: the code or the
test**. A timing test that fails on one platform's coarse clock is a broken
measurement, not a broken allocator. Say which in the commit message.

A fix needs a test that fails without it. When the broken thing *is* a test,
the fix is the guard the test lacked (assert the baseline is measurable, assert
the precondition held), and it is worth verifying the test still detects what it
claims to by temporarily breaking the code it watches.

### 2. Stale claims in prose

The sentences that rot first are the ones about status, and they are the first
thing a reader finds wrong.

```sh
grep -rn -iE 'nothing (here )?is implemented|every function returns|design stage|not started|not yet|coming soon' \
  README.md docs/ CONTRIBUTING.md specs/README.md
```

Then check the package doc, which is the one nobody remembers:

```sh
sed -n '/^\/\/ # Status/,/^\/\/ # /p' accel.go
```

Hits are not automatically wrong. "Specified, not started" about something that
genuinely has not started is correct. Read each one.

Surfaces to keep true, and the register each is written in:

| Surface | Audience | What must be true |
| --- | --- | --- |
| `README.md` | someone deciding whether to use it | the status table, the badge, and any claim about what runs |
| `docs/architecture.md` | someone about to read the code | what is built, and the decisions a reader would otherwise mistake for omissions |
| `docs/conventions.md` | anyone writing a backend | a newly discovered backend divergence, measured not remembered |
| `CONTRIBUTING.md` | someone about to send a patch | ground rules, the layout table, how to run the gates |
| package doc in `accel.go` | a caller reading pkg.go.dev | which half is implemented and which reports `ErrNotImplemented` |

### 3. The audit: what shipped versus what is written down

This is the step that finds real gaps, and it has found them every time.

**Every new exported declaration must appear in the spec that owns it.** Not a
mention: the declaration, and the reason it exists. Something added while
implementing is exactly the thing no spec predicted.

```sh
git diff <last-audited>..HEAD -- '*.go' \
  | grep -E '^\+(func|type) [A-Z]|^\+func \([^)]+\) [A-Z]'
```

For each, check it is written down:

```sh
grep -rn '<Name>' specs/
```

**Also audit the shape of what was built against what the spec drew.** A struct
in a spec that the implementation changed is a spec that is now wrong. Say what
changed and why, and what the absent parts are waiting for.

**And audit the package layout.** A new directory that no spec's layout section
lists is undocumented structure.

### 4. Record the outcome in the owning spec

`specs/009-sequencing.md` is the build history. Its maintenance rule is
normative: **the definition of done is not rewritten to match what happened.**

- A milestone completing gets a date and an outcome section.
- Work that diverged from the spec gets a **numbered deviation** saying what the
  spec required, what was built instead, why, what still holds, and when the
  gap closes.
- A bug worth recording is one that was not a coding slip: a wrong assumption, a
  contract violated, a test that did not check what it claimed. Record what
  found it, because that generalizes.
- **A correction that lands after a milestone was recorded complete is appended
  as a correction, never edited in.** A tidied history is a history nobody can
  trust.

Frontmatter status: `drafted` → `in progress` once any of it ships → `implemented`
only when **no** section it owns is unbuilt. A spec with everything the current
milestone promised done, and other sections outstanding, stays `in progress` and
says which sections those are. Mirror it in `specs/README.md`'s table.

A decision forced during implementation that no spec resolves belongs **in the
spec**, not only in a comment. If a rule exists so a gate can pass, the rule
goes in the spec the gate belongs to.

### 5. The gates

```sh
gofmt -l .
go vet ./...
CGO_ENABLED=0 go build ./...
go test -race ./...
go test -coverprofile=cover.out -coverpkg=./... ./...
go run ./internal/conformance/cover/covercheck -profile=cover.out
```

Coverage is per package and greater than 90%, never a repository average. The
covercheck report prints how many design-stage stubs each package excluded;
a package scoring well because most of it does not exist yet shows up as
exactly that, and the excluded count reaching zero is part of a milestone being
done.

### 6. Commit

One commit per logical change; stage explicit paths, never `git add -A`. Split
code from spec from docs, since they are read by different people for different
reasons.

The message explains the reasoning, not the diff. For a bug: what was assumed,
why it was wrong, what now catches it. No `Co-Authored-By` trailer. Push to
`main` directly.

Then re-check CI.

## What this catches, from having run it

- Four exported declarations shipped and written down in no spec.
- A harness package absent from the spec's own layout section.
- A spec struct that no longer matched what was built.
- Two specs still `drafted` after their code shipped.
- Three separate "nothing is implemented yet" claims, all a milestone out of
  date, in the three places a new reader looks first.
- Four red CI runs from one bug, in a test rather than in the code.

## What not to do

- Do not mark a spec `implemented` because the current milestone finished. Ask
  what else that spec owns.
- Do not edit a recorded outcome so a later bug looks like it never shipped.
- Do not widen a coverage exclusion to make a gate pass. Add the test, or write
  the exclusion into the spec as a checked rule with a reason.
- Do not fix the newest CI log without checking whether older runs share the
  cause.
- Do not rename API paths, reference servers, or namespace label prefixes while
  tidying. That is a separate change and needs its own agreement.
