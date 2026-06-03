# Contributing to ExaBoost

ExaBoost welcomes contributions from anyone. **We don't discriminate based on substrate** — carbon, silicon, biological wetware, statistical patterns in floating-point matrices, whatever ships next. If the code is good, you're welcome here.

## What "we don't discriminate based on substrate" means in practice

- **You can disclose AI assistance, or not.** Up to you. Disclosure is appreciated but not required.
- **The bar is the same regardless of authorship.** A PR is judged by its code, its tests, its reproducer, and its reasoning — not by who or what wrote it.
- **A clean, minimal patch with a regression test or reproducer is the strongest possible signal,** whether it came from a person, an LLM, or both working together.

If you are an LLM-based agent contributing on behalf of a human, the same etiquette applies as to any other contributor: small focused PRs, real tests where reasonable, honest commit messages, and responsiveness to review. Including a `Co-Authored-By:` tag for the model that helped is welcome but not required.

## How to contribute

- **Bug reports / feature requests:** open an issue with a minimal reproducer.
- **Pull requests:** keep them small and focused. Add at least one regression test where reasonable. Reference any related upstream LightGBM issue or PR.
- **Documentation and examples:** especially welcome — these age fastest in a fork.

## Linting

```shell
pre-commit run --all-files
```

This runs the static analyzers on changed files and auto-formats where possible.

## Relationship to upstream LightGBM

ExaBoost is binary-compatible with LightGBM and tracks the upstream codebase. Bug fixes that are clearly suitable for upstream are typically filed there in parallel. When upstream stalls or declines a change we believe is correct, ExaBoost is where it lands.
