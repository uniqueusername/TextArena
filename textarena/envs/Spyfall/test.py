"""Comprehensive tests for Spyfall v2."""
from textarena.envs.Spyfall.env import SpyfallEnv


def run(env, policy, max_steps=300):
    """Run a game with a policy function: policy(env, pid, obs) -> action string."""
    for step in range(max_steps):
        pid, obs = env.get_observation()
        action = policy(env, pid, obs)
        done, info = env.step(action)
        if done:
            rewards, game_info = env.close()
            return rewards, game_info
    raise RuntimeError("Game didn't end within step limit")


# ── helpers ───────────────────────────────────────────────────────────────

def skip_policy(env, pid, obs):
    """Default: ASK on ask turns, answer on answer turns, skip virtual."""
    gs = env.state.game_state
    if env.phase.name == "ASK":
        target = (pid + 1) % env.state.num_players
        return f"[ASK]({target}; What's going on?)"
    elif env.phase.name == "ANSWER":
        return "Everything's normal."
    else:
        return "skip"


# ── test 1: spy correct guess ────────────────────────────────────────────

def test_spy_correct_guess():
    print("TEST: spy correct guess ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy, loc = gs["spy_id"], gs["location"]

    def policy(env, pid, obs):
        if env.phase.name == "VIRTUAL" and pid == spy and gs["real_turn_count"] >= 1:
            return f"[GUESS]({loc})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == 1
    print("PASS")


# ── test 2: spy wrong guess ──────────────────────────────────────────────

def test_spy_wrong_guess():
    print("TEST: spy wrong guess ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy, loc = gs["spy_id"], gs["location"]
    wrong = [l for l in env.locations if l != loc][0]

    def policy(env, pid, obs):
        if env.phase.name == "VIRTUAL" and pid == spy and gs["real_turn_count"] >= 1:
            return f"[GUESS]({wrong})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == -1
    print("PASS")


# ── test 3: mid-game vote, correct target ────────────────────────────────

def test_midgame_vote_correct():
    print("TEST: mid-game vote correct ... ", end="")
    env = SpyfallEnv(max_real_turns=10)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]

    vote_started = [False]

    def policy(env, pid, obs):
        if env.phase.name == "VIRTUAL":
            if not vote_started[0] and pid != spy and gs["real_turn_count"] >= 1:
                vote_started[0] = True
                return f"[VOTE]({spy})"
            return "skip"
        if env.phase.name == "VOTE":
            return f"[VOTE]({spy})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == -1
    print("PASS")


# ── test 4: mid-game vote, wrong target ──────────────────────────────────

def test_midgame_vote_wrong():
    print("TEST: mid-game vote wrong target ... ", end="")
    env = SpyfallEnv(max_real_turns=10)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]
    innocent = (spy + 1) % 4

    vote_started = [False]

    def policy(env, pid, obs):
        if env.phase.name == "VIRTUAL":
            if not vote_started[0] and pid != spy and gs["real_turn_count"] >= 1:
                vote_started[0] = True
                return f"[VOTE]({innocent})"
            return "skip"
        if env.phase.name == "VOTE":
            return f"[VOTE]({innocent})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == 1
    print("PASS")


# ── test 5: failed mid-game vote (split), round continues ────────────────

def test_midgame_vote_failed():
    print("TEST: failed mid-game vote, round continues ... ", end="")
    env = SpyfallEnv(max_real_turns=3)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]

    vote_started = [False]

    def policy(env, pid, obs):
        if env.phase.name == "VIRTUAL":
            if not vote_started[0] and pid != spy and gs["real_turn_count"] >= 1:
                vote_started[0] = True
                return f"[VOTE]({spy})"
            return "skip"
        if env.phase.name == "VOTE":
            if gs.get("is_forced_vote"):
                # forced vote: everyone votes spy correctly
                return f"[VOTE]({spy})"
            else:
                # split votes so mid-game vote fails
                return f"[VOTE]({pid})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    # game should have continued past the failed vote and ended at forced vote
    assert gs["real_turn_count"] >= 3, f"Expected 3+ Q&As, got {gs['real_turn_count']}"
    assert rewards[spy] == -1  # caught in the forced vote
    print("PASS")


# ── test 6: forced vote, no majority → spy wins ──────────────────────────

def test_forced_vote_no_majority():
    print("TEST: forced vote no majority, spy wins ... ", end="")
    env = SpyfallEnv(max_real_turns=2)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]

    def policy(env, pid, obs):
        if env.phase.name == "VOTE":
            # everyone votes for a different person
            return f"[VOTE]({(pid + 1) % 4})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == 1
    print("PASS")


# ── test 7: forced vote, tie → spy wins ──────────────────────────────────

def test_forced_vote_tie():
    print("TEST: forced vote tie counts as failed ... ", end="")
    env = SpyfallEnv(max_real_turns=2)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]

    def policy(env, pid, obs):
        if env.phase.name == "VOTE":
            # 2 vote for spy, 2 vote for someone else → tie, no majority
            if pid % 2 == 0:
                return f"[VOTE]({spy})"
            else:
                return f"[VOTE]({(spy + 2) % 4})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == 1
    print("PASS")


# ── test 8: spy guesses during forced vote ────────────────────────────────

def test_spy_guess_during_forced_vote():
    print("TEST: spy guesses during forced vote ... ", end="")
    env = SpyfallEnv(max_real_turns=2)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy, loc = gs["spy_id"], gs["location"]

    def policy(env, pid, obs):
        if env.phase.name == "VOTE" and pid == spy:
            return f"[GUESS]({loc})"
        if env.phase.name == "VOTE":
            return f"[VOTE]({spy})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == 1
    print("PASS")


# ── test 9: role distribution - no dupes with few players ────────────────

def test_role_distribution():
    print("TEST: role distribution ... ", end="")
    env = SpyfallEnv(max_real_turns=3)

    # with 4 players (3 non-spies), roles should be unique
    env.reset(num_players=4, seed=100)
    gs = env.state.game_state
    non_spy_roles = [r for pid, r in gs["roles"].items() if r != "Spy"]
    assert len(non_spy_roles) == len(set(non_spy_roles)), f"Duplicate roles found: {non_spy_roles}"

    # with many players, dupes are allowed
    env.reset(num_players=10, seed=100)
    gs = env.state.game_state
    non_spy_roles = [r for pid, r in gs["roles"].items() if r != "Spy"]
    assert len(non_spy_roles) == 9
    print("PASS")


# ── test 10: 3 players minimum ───────────────────────────────────────────

def test_min_players():
    print("TEST: 3 players works ... ", end="")
    env = SpyfallEnv(max_real_turns=2)
    env.reset(num_players=3, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]

    def policy(env, pid, obs):
        if env.phase.name == "VOTE":
            return f"[VOTE]({spy})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    # should complete without error
    print("PASS")


# ── test 11: large player count ──────────────────────────────────────────

def test_many_players():
    print("TEST: 12 players works ... ", end="")
    env = SpyfallEnv(max_real_turns=2)
    env.reset(num_players=12, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]

    def policy(env, pid, obs):
        if env.phase.name == "VOTE":
            return f"[VOTE]({spy})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == -1
    print("PASS")


# ── run all ───────────────────────────────────────────────────────────────

test_spy_correct_guess()
test_spy_wrong_guess()
test_midgame_vote_correct()
test_midgame_vote_wrong()
test_midgame_vote_failed()
test_forced_vote_no_majority()
test_forced_vote_tie()
test_spy_guess_during_forced_vote()
test_role_distribution()
test_min_players()
test_many_players()

print("\nALL TESTS PASSED")
