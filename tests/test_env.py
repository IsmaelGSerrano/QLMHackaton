"""load_dotenv reads the repo's .env (found upward) and never overrides the shell."""

from qlmaml.env import load_dotenv


def make_repo(tmp_path, dotenv):
    (tmp_path / ".git").mkdir()  # main checkout: .git is a folder (a worktree has a .git file)
    (tmp_path / ".env").write_text(dotenv)
    worktree = tmp_path / ".claude" / "worktrees" / "x"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: ../../../.git/worktrees/x\n")
    return worktree


def test_worktree_finds_the_main_checkout_dotenv_and_keeps_shell_values(tmp_path):
    worktree = make_repo(tmp_path, '# comment\nIQM_TOKEN="abc123"\n\nNEO4J_PASSWORD=from-file\n')
    environ = {"NEO4J_PASSWORD": "from-shell"}

    found = load_dotenv(worktree, environ=environ)

    assert found == tmp_path / ".env"
    assert environ == {"IQM_TOKEN": "abc123", "NEO4J_PASSWORD": "from-shell"}


def test_inline_comments_are_not_part_of_unquoted_values(tmp_path):
    worktree = make_repo(tmp_path, "IQM_TOKEN=abc123  # resonance\nQUOTED='a # b'\n")
    environ = {}
    load_dotenv(worktree, environ=environ)
    assert environ == {"IQM_TOKEN": "abc123", "QUOTED": "a # b"}


def test_search_stops_at_the_main_checkout_root(tmp_path):
    (tmp_path / ".env").write_text("STRAY=1\n")  # e.g. a .env in the user's home folder
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    environ = {}
    assert load_dotenv(repo, environ=environ) is None
    assert environ == {}
