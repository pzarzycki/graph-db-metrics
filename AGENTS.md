# Project guidelines

- Prioritize simplicity. Choose the simplest solution that meets the requirements; avoid unnecessary abstractions and dependencies.
- Keep commit messages succinct and focused on the change.
- Work directly on `main`; do not create feature branches for this project.
- Use `uv` to manage Python dependencies, environments, and commands.
- Use `npm` to manage web dependencies and scripts.
- Use Neo4j Community Edition for the database; avoid features that require an Enterprise license.
- Write only non-trivial tests that verify meaningful behavior or prevent regressions. Avoid trivial tests and tests that merely mirror the implementation.
- Keep all project data and artifacts within the project directory and its subfolders.
- Maintain appropriate `.gitignore` files as the project evolves. Ignore local data, generated artifacts, caches, environments, installed dependencies, and secrets; keep source files and dependency lockfiles tracked.
