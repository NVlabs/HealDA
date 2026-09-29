Git:
- never run git add -A. carefully avoid checking in binaryies and other data files
- add "Commit message authored by AI" to all git commit messages
- after committing there is no need to summarize changes over again
- before committing, run `make lint`; if it fails, run `make format`

Style:
- do not use docstrings for simple helper functions. signal intent using clear variable and function names.
- comments state facts. do not editorialize, argue with the reader, justify a choice at length, or
  narrate how the code came to be (benchmark campaigns, what was tried, what an earlier revision did).
- no internal identifiers in code or comments: build-system names, campaign ids, solution hashes,
  internal paths or URLs.
- do not duplicate a module or test file per variant. when a feature adds a backend behind a
  toggle, parameterize over the toggle and keep one implementation of the shared logic.

Software enviroment:
- if earth2grid is installed in the global python environment, then run commands as is.
- otherwise run python scripts and tests with uv.

Testing
- Prefer simple tests of the core functionality over exhaustive testing of every possible configuration.
- Avoid complicated pytest code such as hierarchies of inter-dependent fixtures, such code is much better handled using typical python control flow.
- Unit testing for distributed code: `make test-distributed`
