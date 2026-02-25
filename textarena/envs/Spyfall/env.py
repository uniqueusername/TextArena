"""
Spyfall – a social deduction game for 3+ players.

One player is secretly the **spy**; everyone else knows the shared location
and their personal role at that location.  Players take turns asking each
other questions, and between every Q&A pair every player gets a chance to
call a vote or (spy only) guess the location.

Win conditions
──────────────
  Spy wins if:
    • they correctly GUESS the location, OR
    • they remain undetected, OR
    • the non-spies convict the wrong player.

  Non-spies win if:
    • they correctly identify the spy via majority vote, OR
    • the spy guesses the wrong location.

Tools
─────
  All players : [ASK](player_id; question)   – only usable on a real turn
                [VOTE](player_id)            – usable during any vote
  Spy only    : [GUESS](location_name)       – usable on virtual turns and
                                               during the end-of-round vote
                                               (replaces their VOTE)

Turn structure
──────────────
  1.  A random player is chosen to ask first  (real turn – ASK only).
  2.  The recipient answers                   (real turn – free-form text).
  3.  Virtual turn rotation (random order): every player may VOTE or
      GUESS (spy). Anything else is a no-op pass.
  4.  If no game-ending event, the answerer becomes the new asker (goto 1).
  5.  After `max_real_turns` Q&A pairs the round ends.  A forced vote is
      called where every player must VOTE (spy may GUESS instead. If they
      guess correctly, they win. Otherwise, they lose and the non-spies win).
      If there is no majority, the spy wins.

Voting
──────
  • A vote can be initiated on any virtual turn via [VOTE](player_id).
    When this happens, ALL players are polled (random order) to cast votes.
  • A strict majority (> n/2) is required.  Ties count as failed votes.
  • Mid-game failed votes: round continues.
  • End-of-round failed votes: spy wins.
  • During any vote, the spy may [GUESS] instead of voting.
"""

import random
import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.envs.Spyfall.locations import LOCATIONS as DEFAULT_LOCATIONS

# ── game phases ───────────────────────────────────────────────────────────


class Phase(Enum):
    ASK = auto()  # current player must ASK someone
    ANSWER = auto()  # current player answers a question
    VIRTUAL = auto()  # virtual turn: player may VOTE/GUESS or no-op
    VOTE = auto()  # collecting votes from all players


# ── regex patterns ────────────────────────────────────────────────────────

ASK_PATTERN = re.compile(r"\[ASK\]\s*\((\d+)\s*;\s*(.+?)\)", re.IGNORECASE | re.DOTALL)
VOTE_PATTERN = re.compile(r"\[VOTE\]\s*\((\d+)\)", re.IGNORECASE)
GUESS_PATTERN = re.compile(r"\[GUESS\]\s*\((.+?)\)", re.IGNORECASE)


# ── environment ───────────────────────────────────────────────────────────


class SpyfallEnv(ta.Env):
    """
    Spyfall environment for TextArena.

    Parameters
    ----------
    max_real_turns : int
        Maximum number of question-answer pairs before the round ends.
    locations : dict[str, list[str]] | None
        Custom location→roles mapping.  Falls back to the built-in set.
    error_allowance : int
        Number of consecutive invalid moves a player gets before auto-loss.
    """

    def __init__(
        self,
        max_real_turns: int = 8,
        locations: Optional[Dict[str, List[str]]] = None,
        error_allowance: int = 2,
    ):
        self.max_real_turns = max_real_turns
        self.locations = locations or DEFAULT_LOCATIONS
        self.error_allowance = error_allowance

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self, num_players: int, seed: Optional[int] = None):
        assert num_players >= 3, "Spyfall requires at least 3 players."

        self.state = ta.TeamMultiPlayerState(
            num_players=num_players,
            seed=seed,
            error_allowance=self.error_allowance,
        )

        # pick location + assign roles
        location_name = random.choice(list(self.locations.keys()))
        available_roles = list(self.locations[location_name])

        spy_id = random.randint(0, num_players - 1)
        roles: Dict[int, str] = {}

        # distribute roles: no dupes if possible, allow dupes if more
        # players than roles
        non_spy_count = num_players - 1
        if non_spy_count <= len(available_roles):
            sampled = random.sample(available_roles, non_spy_count)
        else:
            # fill unique first, then allow repeats
            sampled = list(available_roles)
            random.shuffle(sampled)
            while len(sampled) < non_spy_count:
                sampled.append(random.choice(available_roles))

        role_idx = 0
        for pid in range(num_players):
            if pid == spy_id:
                roles[pid] = "Spy"
            else:
                roles[pid] = sampled[role_idx]
                role_idx += 1

        # game state
        game_state: Dict[str, Any] = {
            "location": location_name,
            "spy_id": spy_id,
            "roles": roles,
            "real_turn_count": 0,
            "votes": {},
            "answerer_id": None,
            "is_forced_vote": False,  # True during the end-of-round vote
        }

        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._prompt,
            secret_roles={
                pid: ("Spy" if pid == spy_id else "Non-Spy")
                for pid in range(num_players)
            },
        )

        self.phase = Phase.ASK
        self.next_player_ids: List[int] = []
        self._saved_virtual_queue: List[int] = []
        first_asker = random.randint(0, num_players - 1)
        self.state.manually_set_current_player_id(first_asker)

    # ── prompt generation ─────────────────────────────────────────────────

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        n = self.state.num_players
        players_list = ", ".join(f"Player {i}" for i in range(n))
        location_list = ", ".join(sorted(self.locations.keys()))

        base = (
            f"Welcome to Spyfall!  You are Player {player_id}.\n"
            f"Players: {players_list}\n"
            f"Max questions per round: {self.max_real_turns}\n\n"
        )

        if game_state["roles"][player_id] == "Spy":
            base += (
                "You are the SPY.  You do NOT know the location.\n"
                "The other players know the location and each have an assigned role.\n"
                "Your goal: figure out the location without being detected.\n"
                "\n"
                "You can win by:\n"
                "- Guessing the location before the other players discover that you're the spy.\n"
                "- Remaining undetected as the spy for the whole round.\n"
                "- Convincing the remaining players to accuse the wrong player as the spy.\n"
                "\n"
                "The other players can win by:\n"
                "- Accusing you to be the spy via a majority vote.\n"
                "\n"
                "Available tools:\n"
                "  [ASK](player_id; question)  – ask a player a question (your asking turn only)\n"
                "  [VOTE](player_id)           – accuse a player of being the spy (during votes)\n"
                "  [GUESS](location_name)      – guess the location (ends the game immediately!)\n"
                "\n"
                f"Possible locations: {location_list}\n"
            )
        else:
            loc = game_state["location"]
            role = game_state["roles"][player_id]
            base += (
                f"The location is: {loc}\n"
                f"Your role at this location: {role}\n"
                "\n"
                "One of the other players is the SPY (they do NOT know the location).\n"
                "\n"
                "You can win by:\n"
                "- Identifying the spy and accusing them via a majority vote.\n"
                "\n"
                "Available tools:\n"
                "  [ASK](player_id; question)  – ask a player a question (your asking turn only)\n"
                "  [VOTE](player_id)           – accuse a player of being the spy (during votes)\n"
                "\n"
                "The spy can win by:\n"
                "- Guessing the location before you discover who they are.\n"
                "- Remaining undetected for the whole round.\n"
                "- The non-spies accusing the wrong player via a majority vote.\n"
                "\n"
                f"Possible locations: {location_list}\n"
            )

        base += (
            "\nTurn structure:\n"
            "  1. The current asker uses [ASK] to question another player.\n"
            "  2. The recipient answers freely (just type your answer).\n"
            "  3. After each Q&A, every player gets a virtual turn where they\n"
            "     may [VOTE] to accuse someone (or [GUESS] if spy). Anything\n"
            "     else you type is treated as a pass.\n"
            "  4. If a vote is called, ALL players must cast a [VOTE]. The spy\n"
            "     may [GUESS] instead of voting.\n"
            "  5. The answerer becomes the next asker.\n"
            "  6. After all questions are exhausted, a final vote is called.\n"
        )
        return base

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        pid = self.state.current_player_id

        if self.phase == Phase.ASK:
            self._handle_ask(pid, action)
        elif self.phase == Phase.ANSWER:
            self._handle_answer(pid, action)
        elif self.phase == Phase.VIRTUAL:
            self._handle_virtual(pid, action)
        elif self.phase == Phase.VOTE:
            self._handle_vote(pid, action)

        self._advance()
        return self.state.step(rotate_player=False)

    # ── action handlers ───────────────────────────────────────────────────

    def _handle_ask(self, pid: int, action: str):
        m = ASK_PATTERN.search(action)
        if not m:
            self._invalid(
                pid,
                (
                    "You must ask a question using: [ASK](player_id; question). "
                    "Example: [ASK](2; What's the dress code like here?)"
                ),
            )
            return

        target = int(m.group(1))
        question = m.group(2).strip()

        if target == pid:
            self._invalid(pid, "You cannot ask yourself a question.")
            return
        if target < 0 or target >= self.state.num_players:
            self._invalid(pid, f"Player {target} does not exist.")
            return

        self.state.add_observation(
            from_id=pid,
            message=f"Player {pid} asks Player {target}: {question}",
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )
        self.state.game_state["answerer_id"] = target

    def _handle_answer(self, pid: int, action: str):
        self.state.add_observation(
            from_id=pid,
            message=f"Player {pid} answers: {action}",
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )
        self.state.game_state["real_turn_count"] += 1

    def _handle_virtual(self, pid: int, action: str):
        """Virtual turn: VOTE starts a vote, GUESS (spy) ends game, else no-op."""
        gs = self.state.game_state

        # try GUESS (spy only)
        guess_m = GUESS_PATTERN.search(action)
        if guess_m:
            if pid != gs["spy_id"]:
                self._invalid(pid, "Only the spy can use [GUESS].")
                return
            self._resolve_guess(pid, guess_m.group(1).strip())
            return

        # try VOTE — initiates a full vote
        vote_m = VOTE_PATTERN.search(action)
        if vote_m:
            target = int(vote_m.group(1))
            if target < 0 or target >= self.state.num_players:
                self._invalid(pid, f"Player {target} does not exist.")
                return
            # register this player's vote and start collecting from everyone else
            gs["votes"] = {pid: target}
            gs["is_forced_vote"] = False
            self.state.add_observation(
                from_id=pid,
                message=f"Player {pid} initiates a vote to accuse Player {target}! All players must now vote.",
                observation_type=ta.ObservationType.GAME_MESSAGE,
            )
            return

        # anything else = pass (no-op)

    def _handle_vote(self, pid: int, action: str):
        """During a vote: non-spies must VOTE, spy must VOTE or GUESS."""
        gs = self.state.game_state

        # spy may GUESS instead of voting
        guess_m = GUESS_PATTERN.search(action)
        if guess_m:
            if pid != gs["spy_id"]:
                self._invalid(
                    pid, "Only the spy can use [GUESS]. You must [VOTE](player_id)."
                )
                return
            self._resolve_guess(pid, guess_m.group(1).strip())
            return

        # everyone must VOTE
        vote_m = VOTE_PATTERN.search(action)
        if not vote_m:
            self._invalid(pid, "You must vote using [VOTE](player_id).")
            return

        target = int(vote_m.group(1))
        if target < 0 or target >= self.state.num_players:
            self._invalid(pid, f"Player {target} does not exist.")
            return

        gs["votes"][pid] = target
        self.state.add_observation(
            from_id=pid,
            message=f"Player {pid} votes to accuse Player {target}.",
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )

    # ── vote resolution ───────────────────────────────────────────────────

    def _resolve_vote(self):
        """Tally votes and determine outcome."""
        gs = self.state.game_state
        votes = gs["votes"]
        spy_id = gs["spy_id"]
        forced = gs["is_forced_vote"]
        n = self.state.num_players

        # count
        counts: Dict[int, int] = {}
        for target in votes.values():
            counts[target] = counts.get(target, 0) + 1

        # build summary
        vote_summary = ", ".join(
            f"Player {t}: {counts[t]} vote(s)" for t in sorted(counts.keys())
        )
        self.state.add_observation(
            message=f"Vote results — {vote_summary}",
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )

        # majority = strictly more than half
        majority_threshold = n // 2 + 1
        max_count = max(counts.values()) if counts else 0
        top_targets = [t for t, c in counts.items() if c == max_count]

        # tie or no majority → failed vote
        if max_count < majority_threshold or len(top_targets) > 1:
            if forced:
                self._spy_wins(
                    "The vote ended without a clear majority. The spy escapes!"
                )
            else:
                self.state.add_observation(
                    message="No majority reached — the vote fails and the round continues.",
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )
            gs["votes"] = {}
            gs["is_forced_vote"] = False
            return

        # clear majority on a single target
        accused = top_targets[0]
        if accused == spy_id:
            self._nonspies_win(
                f"Player {accused} was correctly identified as the spy! Non-spies win!"
            )
        else:
            self._spy_wins(
                f"Player {accused} was accused, but they are NOT the spy! The spy wins!"
            )

    # ── guess resolution ──────────────────────────────────────────────────

    def _resolve_guess(self, pid: int, guessed_location: str):
        gs = self.state.game_state
        self.state.add_observation(
            from_id=pid,
            message=f"Player {pid} reveals they are the spy and guesses the location: {guessed_location}",
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )

        if guessed_location.strip().lower() == gs["location"].strip().lower():
            self._spy_wins(
                f"The spy correctly guessed the location ({gs['location']})! Spy wins!"
            )
        else:
            self._nonspies_win(
                f"The spy guessed '{guessed_location}', but the real location is "
                f"'{gs['location']}'. Non-spies win!"
            )

    # ── advance / turn management ─────────────────────────────────────────

    def _advance(self):
        """After every action, decide what happens next."""
        if self.state.made_invalid_move:
            return
        if self.state.done:
            return

        gs = self.state.game_state

        # if a vote was just initiated from a virtual turn, transition to VOTE
        # and queue ALL other players (random order)
        if self.phase == Phase.VIRTUAL and gs["votes"]:
            self._save_virtual_queue()
            self._start_vote_collection()
            return

        # if there are still queued players, rotate
        if self.next_player_ids:
            self.state.manually_set_current_player_id(self.next_player_ids.pop())
            return

        # queue exhausted — resolve current phase and transition

        if self.phase == Phase.ASK:
            self.phase = Phase.ANSWER
            self.state.manually_set_current_player_id(gs["answerer_id"])
            return

        if self.phase == Phase.ANSWER:
            self._start_virtual_rotation()
            return

        if self.phase == Phase.VIRTUAL:
            # all virtual turns done, no vote was called
            if gs["real_turn_count"] >= self.max_real_turns:
                self._start_forced_vote()
                return
            # next Q&A
            self.phase = Phase.ASK
            self.state.manually_set_current_player_id(gs["answerer_id"])
            return

        if self.phase == Phase.VOTE:
            # all votes collected — resolve
            self._resolve_vote()
            if self.state.done:
                return
            # vote failed — resume
            self._restore_after_vote()
            return

    def _start_virtual_rotation(self):
        """Queue every player for a virtual turn (random order)."""
        self.phase = Phase.VIRTUAL
        n = self.state.num_players
        order = list(range(n))
        random.shuffle(order)
        self.next_player_ids = list(reversed(order))  # reversed for pop()

        spy_id = self.state.game_state["spy_id"]
        for pid in order:
            if pid == spy_id:
                self.state.add_observation(
                    to_id=pid,
                    message=(
                        "Virtual turn: you may [VOTE](player_id) to accuse someone, "
                        "[GUESS](location) to guess the location, or say anything else to pass."
                    ),
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )
            else:
                self.state.add_observation(
                    to_id=pid,
                    message=(
                        "Virtual turn: you may [VOTE](player_id) to accuse someone, "
                        "or say anything else to pass."
                    ),
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )

        self.state.manually_set_current_player_id(self.next_player_ids.pop())

    def _save_virtual_queue(self):
        """Save remaining virtual turns before switching to vote phase."""
        self._saved_virtual_queue = list(self.next_player_ids)
        self.next_player_ids = []

    def _start_vote_collection(self):
        """Switch to VOTE phase and queue all players who haven't voted."""
        self.phase = Phase.VOTE
        gs = self.state.game_state
        already_voted = set(gs["votes"].keys())
        remaining = [
            pid for pid in range(self.state.num_players) if pid not in already_voted
        ]
        random.shuffle(remaining)
        self.next_player_ids = list(reversed(remaining))

        spy_id = gs["spy_id"]
        for pid in remaining:
            if pid == spy_id:
                self.state.add_observation(
                    to_id=pid,
                    message="A vote has been called! You must [VOTE](player_id) or [GUESS](location).",
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )
            else:
                self.state.add_observation(
                    to_id=pid,
                    message="A vote has been called! You must [VOTE](player_id).",
                    observation_type=ta.ObservationType.GAME_MESSAGE,
                )

        self.state.manually_set_current_player_id(self.next_player_ids.pop())

    def _start_forced_vote(self):
        """End-of-round: start a forced vote (all players, random order).

        Looks identical to a normal vote from every player's perspective.
        The spy's prompt tells them they may GUESS instead.  Non-spies
        don't know this is the "final" vote — no spy reveal occurs.
        """
        gs = self.state.game_state
        gs["votes"] = {}
        gs["is_forced_vote"] = True
        self._saved_virtual_queue = []  # nothing to resume after

        self.phase = Phase.VOTE
        n = self.state.num_players
        order = list(range(n))
        random.shuffle(order)
        self.next_player_ids = list(reversed(order))

        spy_id = gs["spy_id"]
        self.state.add_observation(
            message=(
                f"The round has ended after {self.max_real_turns} questions. "
                "A final vote is now called — every player must vote!"
            ),
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )
        self.state.add_observation(
            to_id=spy_id,
            message="You may [GUESS](location) instead of voting to attempt to win.",
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )

        self.state.manually_set_current_player_id(self.next_player_ids.pop())

    def _restore_after_vote(self):
        """After a failed mid-game vote, resume virtual turns or next Q&A."""
        saved = list(self._saved_virtual_queue)
        self._saved_virtual_queue = []

        if saved:
            # resume remaining virtual turns
            self.phase = Phase.VIRTUAL
            self.next_player_ids = saved
            self.state.manually_set_current_player_id(self.next_player_ids.pop())
            return

        gs = self.state.game_state
        # check if round ended during the vote collection
        if gs["real_turn_count"] >= self.max_real_turns:
            self._start_forced_vote()
            return

        # next Q&A
        self.phase = Phase.ASK
        self.state.manually_set_current_player_id(gs["answerer_id"])

    # ── outcome helpers ───────────────────────────────────────────────────

    def _spy_wins(self, reason: str):
        spy_id = self.state.game_state["spy_id"]
        self.state.add_observation(
            message=reason,
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )
        self.state.set_winners(player_ids=[spy_id], reason=reason)

    def _nonspies_win(self, reason: str):
        spy_id = self.state.game_state["spy_id"]
        non_spies = [pid for pid in range(self.state.num_players) if pid != spy_id]
        self.state.add_observation(
            message=reason,
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )
        self.state.set_winners(player_ids=non_spies, reason=reason)

    def _invalid(self, pid: int, reason: str):
        fatal = self.state.set_invalid_move(reason)
        if fatal:
            gs = self.state.game_state
            if pid == gs["spy_id"]:
                non_spies = [p for p in range(self.state.num_players) if p != pid]
                self.state.set_winners(
                    player_ids=non_spies,
                    reason=f"Player {pid} (spy) eliminated for repeated invalid moves.",
                )
            else:
                self.state.set_winners(
                    player_ids=[gs["spy_id"]],
                    reason=f"Player {pid} eliminated for repeated invalid moves.",
                )
            self.state.made_invalid_move = False
