# Contributing to NVFlow

Thank you for your interest in contributing to NVFlow.

## Signing Your Work

We require that all contributors **sign off** on their commits. This certifies that the contribution is your original work, or that you have the right to submit it under the same license (or a compatible license).

**Any contribution that contains commits that are not signed off will not be accepted.**

### How to sign off

Use the `--signoff` (or `-s`) option when committing your changes:

```bash
git commit -s -m "Add cool feature."
```

This will append a line to your commit message, for example:

```
Signed-off-by: Your Name <your.email@example.com>
```

### Full text of the Developer Certificate of Origin (DCO)

The sign-off certifies compliance with the Developer Certificate of Origin (DCO). Full text: [https://developercertificate.org/](https://developercertificate.org/)

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have
    the right to submit it under the open source license indicated in
    the file; or

(b) The contribution is based upon previous work that, to the best of
    my knowledge, is covered under an appropriate open source license
    and I have the right under that license to submit that work with
    modifications, whether created in whole or in part by me, under
    the same open source license (unless I am permitted to submit under
    a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person
    who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are
    public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Proposing Major Changes

Discuss a significant change in an RFC issue before opening a pull request. See [Requests for comments](docs/rfcs/README.md).

## Pull Requests

1. Fork the repository and create a branch from `main`.
2. Make your changes with signed-off commits (`git commit -s`).
3. Push your branch and open a Pull Request into `main`.
4. Ensure the PR description references any related issues and describes the change clearly.

## Code and Documentation

- Follow existing code style and conventions in the project.
- Before submitting, run the checks CI runs. With the README's [contributor setup](README.md#1-install-the-local-cli) installed, `uv run pre-commit run --all-files` runs the linters.
- `scripts/run_ci_test_lane.sh minimal` runs the tests that need no NeMo-Skills, NeMo-Run, or Torch, and `scripts/run_ci_test_lane.sh nemo` runs the tests that need them. The `minimal` lane leaves only its CI dependencies installed, so rerun the contributor setup after it.
- Update documentation if you change behavior or add features.

## Dependency Updates

Runtime and CI tooling dependencies share one `uv.lock`; do not create a second
CI requirements lock. Prefer a narrow update so unrelated packages stay fixed:

```bash
uv lock --upgrade-package <package>
uv lock --check
```

Commit `pyproject.toml` and `uv.lock` together, and check that the update adds
no fixable HIGH or CRITICAL vulnerabilities:

```bash
trivy fs --scanners vuln --include-dev-deps --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 .
```

New or updated third-party dependencies also require the license review under
[Third-Party Code](#third-party-code).

## License

By contributing, you agree that your contributions will be licensed under the same license as the project: the Apache License, Version 2.0. See [LICENSE](LICENSES/LICENSE) for the full text.

## Third-Party Code

If your contribution adds or updates a third-party dependency, make sure its license is compatible with Apache 2.0 and record it in [`LICENSES/THIRD_PARTY_SW_LICENSE_INFO.md`](LICENSES/THIRD_PARTY_SW_LICENSE_INFO.md).
