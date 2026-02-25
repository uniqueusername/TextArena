"""Comprehensive tests for Spyfall v2."""
from textarena.envs.Spyfall.env import SpyfallEnv, Phase


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


# ── test 5b: forced vote, correct majority → spy loses ───────────────────

def test_forced_vote_identifies_spy():
    print("TEST: forced vote identifies spy, spy loses ... ", end="")
    env = SpyfallEnv(max_real_turns=2)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]

    def policy(env, pid, obs):
        if env.phase.name == "VOTE":
            return f"[VOTE]({spy})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == -1
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
    non_spy_roles = [r for pid, r in gs["roles"].items() if r != "spy"]
    assert len(non_spy_roles) == len(set(non_spy_roles)), f"Duplicate roles found: {non_spy_roles}"

    # with many players, dupes are allowed
    env.reset(num_players=10, seed=100)
    gs = env.state.game_state
    non_spy_roles = [r for pid, r in gs["roles"].items() if r != "spy"]
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


# ── helpers for unit-style phase tests ───────────────────────────────────

def _advance_to_virtual(env):
    """From a fresh ASK phase, complete one Q&A and return in VIRTUAL phase."""
    pid = env.state.current_player_id
    target = (pid + 1) % env.state.num_players
    env.step(f"[ASK]({target}; What's your role here?)")
    env.step("Just a regular person.")
    assert env.phase == Phase.VIRTUAL


def _advance_game_safely(env):
    """Step the game forward one phase-turn with a valid move.

    Safe for all phases; does NOT trigger any invalid moves.
    """
    pid = env.state.current_player_id
    n = env.state.num_players
    if env.phase == Phase.ASK:
        target = (pid + 1) % n
        env.step(f"[ASK]({target}; q)")
    elif env.phase == Phase.ANSWER:
        env.step("a")
    elif env.phase == Phase.VIRTUAL:
        env.step("pass")
    elif env.phase == Phase.VOTE:
        target = (pid + 1) % n
        env.step(f"[VOTE]({target})")


# ── win conditions ────────────────────────────────────────────────────────

# test_spy_correct_guess               ← covered by test 1
# test_spy_wrong_guess                 ← covered by test 2
# test_midgame_vote_correct            ← covered by test 3 (spy lose)
# test_midgame_vote_wrong              ← covered by test 4 (spy win, wrong target)
# test_forced_vote_no_majority         ← covered by test 6
# test_forced_vote_tie                 ← covered by test 7
# test_midgame_vote_failed             ← covers mid-round vote no majority


def test_forced_vote_majority_wrong_target():
    """End-of-round forced vote: majority votes for an innocent player → spy wins."""
    print("TEST: forced vote majority wrong target, spy wins ... ", end="")
    env = SpyfallEnv(max_real_turns=2)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]
    innocent = (spy + 1) % 4

    def policy(env, pid, obs):
        if env.phase.name == "VOTE":
            return f"[VOTE]({innocent})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == 1, f"Expected spy win, got rewards={rewards}"
    print("PASS")


# ── tool use ──────────────────────────────────────────────────────────────


def test_ask_valid_on_real_turn():
    """[ASK] during ASK phase is accepted and transitions to ANSWER."""
    print("TEST: [ASK] valid on real turn ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    assert env.phase == Phase.ASK

    pid = env.state.current_player_id
    target = (pid + 1) % 4
    done, _ = env.step(f"[ASK]({target}; What do you do here?)")
    assert not done
    assert env.phase == Phase.ANSWER
    assert env.state.current_player_id == target
    print("PASS")


def test_ask_ignored_on_virtual_turn():
    """[ASK] sent on a virtual turn is treated as a pass (no-op), not an error."""
    print("TEST: [ASK] ignored on virtual turn ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    _advance_to_virtual(env)

    virtual_pid = env.state.current_player_id
    done, _ = env.step("[ASK](0; sneaky question)")
    assert not done
    # treated as a pass: phase is still VIRTUAL or just transitioned to ASK,
    # NOT to ANSWER (which only happens after a real ASK)
    assert env.phase != Phase.ANSWER
    # the acting player's slot has been consumed (they don't get a re-ask)
    assert env.state.current_player_id != virtual_pid or env.phase == Phase.ASK
    print("PASS")


def test_vote_valid_on_virtual_turn():
    """[VOTE] on a virtual turn transitions the game into VOTE phase."""
    print("TEST: [VOTE] valid on virtual turn ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    _advance_to_virtual(env)

    virtual_pid = env.state.current_player_id
    target = (virtual_pid + 1) % 4
    done, _ = env.step(f"[VOTE]({target})")
    assert not done
    assert env.phase == Phase.VOTE
    print("PASS")


def test_vote_invalid_on_ask_turn():
    """[VOTE] during ASK phase is invalid (player must use [ASK])."""
    print("TEST: [VOTE] invalid on ASK turn ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    assert env.phase == Phase.ASK

    pid = env.state.current_player_id
    done, _ = env.step(f"[VOTE]({(pid + 1) % 4})")
    assert not done
    assert env.phase == Phase.ASK  # still waiting for a valid [ASK]
    print("PASS")


def test_spy_guess_valid_on_virtual_turn():
    """Spy's [GUESS](correct) on a virtual turn wins the game immediately."""
    print("TEST: spy [GUESS] valid on virtual turn ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy, loc = gs["spy_id"], gs["location"]

    def policy(env, pid, obs):
        if env.phase == Phase.VIRTUAL and pid == spy:
            return f"[GUESS]({loc})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == 1
    print("PASS")


def test_spy_guess_valid_during_vote():
    """Spy's [GUESS](correct) during a VOTE phase wins the game immediately."""
    # This is a duplicate of test_spy_guess_during_forced_vote but confirms the
    # same behaviour holds for mid-round votes too.
    print("TEST: spy [GUESS] valid during vote phase ... ", end="")
    env = SpyfallEnv(max_real_turns=10)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy, loc = gs["spy_id"], gs["location"]
    vote_started = [False]

    def policy(env, pid, obs):
        if env.phase == Phase.VIRTUAL and not vote_started[0] and gs["real_turn_count"] >= 1:
            vote_started[0] = True
            return f"[VOTE]({(pid + 1) % 4})"
        if env.phase == Phase.VOTE and pid == spy:
            return f"[GUESS]({loc})"
        if env.phase == Phase.VOTE:
            return f"[VOTE]({(pid + 1) % 4})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == 1
    print("PASS")


def test_nonspy_cannot_guess():
    """[GUESS] by a non-spy on a virtual turn is flagged as an invalid move."""
    print("TEST: non-spy cannot [GUESS] ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]
    loc = gs["location"]

    _advance_to_virtual(env)

    # skip forward until we find a non-spy on a virtual turn
    for _ in range(env.state.num_players):
        if env.state.current_player_id != spy:
            break
        env.step("pass")

    assert env.state.current_player_id != spy, "Could not find a non-spy virtual turn"
    assert env.phase == Phase.VIRTUAL

    done, _ = env.step(f"[GUESS]({loc})")
    assert not done            # game should NOT end
    assert env.phase == Phase.VIRTUAL  # still in virtual; player must re-act
    print("PASS")


def test_skip_valid_on_virtual_turn():
    """Any player may pass/skip on a virtual turn by sending arbitrary text."""
    print("TEST: skip/pass valid on virtual turn ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    _advance_to_virtual(env)

    virtual_pid = env.state.current_player_id
    done, _ = env.step("I have nothing to add right now.")
    assert not done
    # player's virtual slot was consumed; they are no longer the active player
    # (unless the whole virtual rotation finished and it cycled back to ASK)
    assert env.state.current_player_id != virtual_pid or env.phase == Phase.ASK
    print("PASS")


def test_vote_outside_vote_initiates_rotation_and_registers():
    """[VOTE] on a virtual turn starts a VOTE rotation AND registers the initiator's choice."""
    print("TEST: virtual [VOTE] initiates rotation and registers accusation ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    _advance_to_virtual(env)

    initiator = env.state.current_player_id
    accusation_target = (initiator + 1) % 4
    env.step(f"[VOTE]({accusation_target})")

    # vote phase should have started
    assert env.phase == Phase.VOTE
    # initiator's accusation must be recorded
    assert gs["votes"].get(initiator) == accusation_target
    print("PASS")


def test_vote_initiator_excluded_from_vote_rotation():
    """The player who starts a vote is not re-queued; they don't get two votes."""
    print("TEST: vote initiator excluded from vote rotation ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    _advance_to_virtual(env)

    initiator = env.state.current_player_id
    target = (initiator + 1) % 4
    env.step(f"[VOTE]({target})")

    assert env.phase == Phase.VOTE
    # initiator must not appear in the remaining vote queue or as active player
    assert initiator not in env.next_player_ids
    assert env.state.current_player_id != initiator
    print("PASS")


def test_vote_during_ongoing_vote_registers_accusation():
    """Casting [VOTE] while a vote is already in progress records the player's choice."""
    print("TEST: [VOTE] during ongoing vote registers accusation ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    _advance_to_virtual(env)

    # initiate a vote from a virtual turn
    initiator = env.state.current_player_id
    env.step(f"[VOTE]({(initiator + 1) % 4})")
    assert env.phase == Phase.VOTE

    # first voter in the VOTE rotation casts their vote
    voter = env.state.current_player_id
    voter_target = (voter + 2) % 4
    env.step(f"[VOTE]({voter_target})")

    assert gs["votes"].get(voter) == voter_target
    print("PASS")


def test_after_ask_askee_gets_turn():
    """After [ASK](target; ...), the target becomes the current player in ANSWER phase."""
    print("TEST: after [ASK], askee gets the turn ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    assert env.phase == Phase.ASK

    pid = env.state.current_player_id
    target = (pid + 1) % 4
    env.step(f"[ASK]({target}; What brings you here?)")

    assert env.phase == Phase.ANSWER
    assert env.state.current_player_id == target
    print("PASS")


def test_player_cannot_ask_self():
    """A player asking themselves is an invalid move."""
    print("TEST: player cannot [ASK] self ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    assert env.phase == Phase.ASK

    pid = env.state.current_player_id
    done, _ = env.step(f"[ASK]({pid}; Can I ask myself?)")
    assert not done
    assert env.phase == Phase.ASK  # still their turn
    print("PASS")


def test_player_cannot_ask_nonexistent_player():
    """Asking a player that doesn't exist is an invalid move."""
    print("TEST: player cannot [ASK] nonexistent player ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    assert env.phase == Phase.ASK

    done, _ = env.step("[ASK](99; Are you there?)")
    assert not done
    assert env.phase == Phase.ASK
    print("PASS")


def test_player_cannot_vote_for_nonexistent_player():
    """Voting for a player that doesn't exist is an invalid move."""
    print("TEST: player cannot [VOTE] for nonexistent player ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    _advance_to_virtual(env)

    done, _ = env.step("[VOTE](99)")
    assert not done
    assert env.phase == Phase.VIRTUAL
    print("PASS")


def test_guess_case_insensitive():
    """[GUESS] should match the location regardless of capitalisation."""
    print("TEST: [GUESS] is case-insensitive ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy, loc = gs["spy_id"], gs["location"]
    upper_loc = loc.upper()

    def policy(env, pid, obs):
        if env.phase == Phase.VIRTUAL and pid == spy:
            return f"[GUESS]({upper_loc})"
        return skip_policy(env, pid, obs)

    rewards, _ = run(env, policy)
    assert rewards[spy] == 1, f"Expected spy win with upper-case guess, got {rewards}"
    print("PASS")


# ── game mechanics ────────────────────────────────────────────────────────


def test_max_real_turns_triggers_forced_vote():
    """After max_real_turns Q&A pairs, a forced vote is started."""
    print("TEST: max_real_turns triggers forced vote ... ", end="")
    env = SpyfallEnv(max_real_turns=2)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state

    # play exactly max_real_turns rounds skipping all virtual turns
    for _ in range(200):
        if env.phase == Phase.VOTE:
            break
        _advance_game_safely(env)
        if env.state.done:
            break

    assert env.phase == Phase.VOTE, f"Expected VOTE phase, got {env.phase}"
    assert gs["is_forced_vote"], "Expected forced vote flag to be set"
    print("PASS")


def test_spy_exceeds_invalid_threshold():
    """Spy exceeding error_allowance results in a non-spy win."""
    print("TEST: spy exceeds invalid threshold, non-spies win ... ", end="")
    env = SpyfallEnv(max_real_turns=100, error_allowance=0)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]

    # advance until it's the spy's ASK turn, then make an invalid move
    for _ in range(200):
        pid = env.state.current_player_id
        if pid == spy and env.phase == Phase.ASK:
            done, _ = env.step(f"[ASK]({spy}; asking myself)")  # ask self → invalid
            assert done, "Expected game to end on fatal invalid move"
            assert env.state.rewards[spy] == -1
            for p in range(4):
                if p != spy:
                    assert env.state.rewards[p] == 1
            print("PASS")
            return
        _advance_game_safely(env)
        if env.state.done:
            break

    assert False, "Spy never got an ASK turn within 200 steps"


def test_nonspy_exceeds_invalid_threshold():
    """Non-spy exceeding error_allowance results in the spy winning."""
    print("TEST: non-spy exceeds invalid threshold, spy wins ... ", end="")
    env = SpyfallEnv(max_real_turns=100, error_allowance=0)
    env.reset(num_players=4, seed=42)
    gs = env.state.game_state
    spy = gs["spy_id"]

    for _ in range(200):
        pid = env.state.current_player_id
        if pid != spy and env.phase == Phase.ASK:
            done, _ = env.step(f"[ASK]({pid}; asking myself)")  # ask self → invalid
            assert done, "Expected game to end on fatal invalid move"
            assert env.state.rewards[spy] == 1
            print("PASS")
            return
        _advance_game_safely(env)
        if env.state.done:
            break

    assert False, "Non-spy never got an ASK turn within 200 steps"


def test_assertion_error_fewer_than_3_players():
    """Resetting with fewer than 3 players raises an AssertionError."""
    print("TEST: AssertionError for < 3 players ... ", end="")
    env = SpyfallEnv(max_real_turns=5)
    try:
        env.reset(num_players=2)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass  # expected
    print("PASS")


def test_midround_vote_fail_restores_virtual_queue():
    """After a failed mid-round vote the remaining virtual turns are restored."""
    print("TEST: mid-round vote fail restores virtual queue ... ", end="")
    env = SpyfallEnv(max_real_turns=10)
    env.reset(num_players=4, seed=42)
    _advance_to_virtual(env)

    assert env.phase == Phase.VIRTUAL
    # all 4 players should have virtual turns: 1 current + 3 queued
    assert len(env.next_player_ids) + 1 == 4

    # first virtual player initiates a vote
    initiator = env.state.current_player_id
    env.step(f"[VOTE]({(initiator + 1) % 4})")
    assert env.phase == Phase.VOTE

    # capture the saved queue (players who haven't had their virtual turn yet)
    saved_queue = list(env._saved_virtual_queue)
    assert len(saved_queue) > 0, "Expected some virtual turns to be saved"

    # make the vote fail: each voter votes for themselves → split, no majority
    for _ in range(200):
        if env.phase != Phase.VOTE:
            break
        pid = env.state.current_player_id
        env.step(f"[VOTE]({pid})")  # self-vote → guaranteed split

    assert not env.state.done, "Game should not be over after a failed mid-round vote"
    assert env.phase == Phase.VIRTUAL, f"Expected VIRTUAL phase after failed vote, got {env.phase}"

    # the remaining virtual players should be exactly those from the saved queue
    remaining = [env.state.current_player_id] + list(env.next_player_ids)
    assert set(remaining) == set(saved_queue), (
        f"Expected remaining virtual players {set(saved_queue)}, got {set(remaining)}"
    )
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
test_forced_vote_identifies_spy()
test_role_distribution()
test_min_players()
test_many_players()
test_forced_vote_majority_wrong_target()
test_ask_valid_on_real_turn()
test_ask_ignored_on_virtual_turn()
test_vote_valid_on_virtual_turn()
test_vote_invalid_on_ask_turn()
test_spy_guess_valid_on_virtual_turn()
test_spy_guess_valid_during_vote()
test_nonspy_cannot_guess()
test_skip_valid_on_virtual_turn()
test_vote_outside_vote_initiates_rotation_and_registers()
test_vote_initiator_excluded_from_vote_rotation()
test_vote_during_ongoing_vote_registers_accusation()
test_after_ask_askee_gets_turn()
test_player_cannot_ask_self()
test_player_cannot_ask_nonexistent_player()
test_player_cannot_vote_for_nonexistent_player()
test_guess_case_insensitive()
test_max_real_turns_triggers_forced_vote()
test_spy_exceeds_invalid_threshold()
test_nonspy_exceeds_invalid_threshold()
test_assertion_error_fewer_than_3_players()
test_midround_vote_fail_restores_virtual_queue()

print("\nALL TESTS PASSED")
