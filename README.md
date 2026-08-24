# Tengine-musa

Tengine-musa is a Moore Threads maintained MUSA adaptation of Tengine for neural-network inference on front-end and embedded devices. The repository keeps the upstream project structure and adds MUSA-oriented source, build, and integration changes on top of the inherited upstream baseline.

## Upstream And Version

- Upstream project: https://github.com/OAID/Tengine
- Inherited source version: v1.5
- Upstream baseline commit: `5ec1c383c8adb0078c025b9fec6fa3dea254034a`
- MUSA migration ref used for this publication: `origin/musa-Tengine`
- Publication branch: `musa-v1.5`

## MUSA Adaptation

This branch carries Moore Threads changes for MUSA enablement while preserving the upstream source layout and attribution. The adaptation may include MUSA backend integration, kernel or runtime compatibility updates, build-system wiring, and documentation changes needed for a public MUSA migration repository.

The repository intentionally retains inherited upstream source, examples, tests, and documentation unless they are superseded by this public MUSA note. Upstream documentation remains useful for project concepts and non-MUSA workflows.

## Dependencies

Install the toolchains and package dependencies required by the upstream project, then add the Moore Threads MUSA software stack required by the MUSA-enabled paths in this branch. Dependency versions and optional features are governed by the retained upstream build files in this repository.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Adjust compiler paths, MUSA toolkit paths, and optional backend flags for your local environment. These commands are source-level entry points only; no runtime status is implied by this publication note.

## Basic Usage

Use the upstream command-line tools, Python modules, libraries, or examples as documented by the inherited project. For MUSA-specific execution, select the MUSA-enabled build or runtime path provided by this branch and follow the relevant upstream workflow for inputs and outputs.

## Repository Layout

- `AGENTS.md`
- `CMakeLists.txt`
- `LICENSE`
- `MIGRATION_REPORT.md`
- `README.md`
- `README_EN.md`
- `benchmark`
- `cmake`
- `demos`
- `doc`
- `examples`
- `logo-Tengine.png`
- `pytengine`
- `scripts`

## Contributing

Contributions should preserve upstream attribution, keep MUSA-specific changes clearly scoped, and avoid removing inherited functionality without a source-compatible replacement. Please include concise build or usage notes for changes that affect the MUSA path.

## Attribution And License

This repository is derived from the upstream project listed above. Moore Threads maintains the MUSA adaptation in this branch. See [LICENSE](LICENSE) for the root copyright and license statement used for this publication.
