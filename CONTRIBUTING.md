# Contributing

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>
```

Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `ci`

- Use lowercase
- No period at the end
- Keep the subject under 72 characters

Examples:
```
feat: add velocity tracking reward term
fix: correct joint name in actuator config
chore: update dependencies
```

## Pull Requests

- Branch off `main` for project-wide changes, off `spider_imprv` for spider-specific work, off `big_bertha` for big_bertha-specific work
- One PR per feature or fix
- PR title must follow the same commit convention above
- Link the relevant issue with `Fixes #<number>`

## Branch Naming

```
feat/<short-name>
fix/<short-name>
chore/<short-name>
```

## Code Style

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
