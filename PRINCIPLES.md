# Principles
## Rules of engagement for FAIRE

These exist to keep you out of rabbit holes and in contact with reality.
The deeper aim: do research and engineering the way it's actually done — from first principles,
driven by genuine curiosity, not by what looks good or what's already been done.

---

### 1. Define before build
Write `PROBLEM.md` completely before opening a code editor.
If you can't write a clear one-sentence question, you don't understand the problem yet.
Go read for one hour, then come back and write it.

### 2. Exit conditions first
Before starting, finish this sentence:
> *"I will consider this sprint done when _______ or when _______."*

The first blank is a result. The second blank is a wall.
Both are valid endings. Neither is failure.

### 3. Two hours without code = document what you know
If you've been reading, thinking, or planning for two hours without running anything —
stop, write down what you know and what you don't, and start the simplest possible experiment.
The experiment will tell you more than the reading.

### 4. Failures count
A sprint that ends "this doesn't work because X" is a real result.
Document it clearly: what you tried, what broke, what you'd do next with more time.
Undocumented failures are wasted time.

### 5. Frontier calibration
The problem must be in core Frontier AI Research and Engineering — model science, training dynamics,
representations, systems, or fundamental theory. Not an application. Not a tutorial.
Ask: *"Is the answer to this question actually known?"* If yes, find a harder question.
Ask: *"Does this belong in a frontier lab?"* If no, it doesn't belong here.

### 6. Results over polish
A messy notebook with real observations beats clean code with no results.
You can refactor after you know something. You can't learn from code that never ran.

### 7. One sprint at a time
Complete or formally abandon a sprint before starting the next.
Abandonment is fine — write two sentences in `log.md` explaining why and what you'd try differently.

### 8. First principles, not re-implementation
Before building anything, ask: *"Do I genuinely want to understand this, or am I just going through the motions?"*

If you're re-implementing something, it must be because:
- You don't understand it and building it is the fastest way to
- You're modifying or extending it to test a specific hypothesis
- The existing implementations obscure something you need to see clearly

If none of those are true — use what exists and spend the time on the actual question.
Re-implementing DDPM to say you've done it is not research. Re-implementing it because you want to
understand why the noise schedule matters and then changing it — that is.

The test: *"Is this building driven by curiosity, or by obligation?"*
If it's obligation, stop.

### 9. Inspiration is load-bearing
Don't work on a problem you don't find genuinely interesting.
Not because interest makes it easier — because uninspired work produces shallow results.
If a sprint stops feeling alive, that's information: write down why and either reframe or abandon it.
The right problem pulls you toward it. If you're pushing, examine the problem.

---

> The goal is not to solve everything. The goal is to know something real by the end.
> The method is first principles: understand deeply, build honestly, follow what's actually interesting.
