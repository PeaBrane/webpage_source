# Repository Workflow

- This is a Hugo source repository. Edit source files under `content/`, `layouts/`, `assets/`, `static/`, and `config.toml`.
- `themes/PaperMod` is the Hugo theme submodule. Do not edit or advance it unless the task explicitly requires a theme change.
- `public` is the generated-site submodule for `PeaBrane/peabrane.github.io`. Treat its contents as generated output; do not edit them by hand.
- Initialize or refresh submodules after cloning or pulling:
  ```bash
  git pull --recurse-submodules
  git submodule update --init --recursive
  ```
- Preview source changes locally with `hugo server -D`, then inspect the site at `http://localhost:1313/`.
- Build production output with `hugo --minify`. This writes the generated site into `public/`.
- The checked-in generated site was last built with Hugo 0.155.3. Check the installed Hugo version before rebuilding and review unexpected generated-file churn caused by version drift.
- Before committing, inspect both repositories:
  ```bash
  git status --short
  git diff
  git -C public status --short
  git -C public diff --stat
  ```
- Commit and push `public` first so the outer repository can record the new generated-site commit:
  ```bash
  git -C public add -A
  git -C public commit -s -m "deploy: describe change"
  git -C public push origin main
  ```
- After pushing `public`, stage the intended source files and the updated `public` submodule pointer, then commit and push the outer repository:
  ```bash
  git add <source-files> public
  git commit -s -m "describe change"
  git push origin main
  ```
- Pushing `public/main` publishes the GitHub Pages site. Pushing the outer repository preserves the source changes and the deployed submodule revision.
- Preserve unrelated local changes. Never reset, clean, or overwrite either repository to resolve unexpected differences.
