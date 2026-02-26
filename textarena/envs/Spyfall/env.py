"""
spyfall (3+ players)

one player is secretly the **spy**; everyone else knows the shared location
and their personal role at that location. players take turns asking each
other questions, and between every Q&A pair every player gets a chance to
call a vote or (spy only) guess the location.

game rules
  spy wins if
    • they correctly guess the location, OR
    • they remain undetected, OR
    • the non-spies convict the wrong player.

  non-spies win if
    • they correctly identify the spy via majority vote, OR
    • the spy guesses the wrong location.

functional details
  • a "real turn" involves the player whose turn it is asking another
    player a question, and the askee answering the question.
  • as any player can call a vote (or the spy can guess) at any time,
    every "real turn" is part of a "virtual turn". after a question
    is answered, we pass the turn through every player in a random
    order so each one has a chance to vote or guess.
  • real games of spyfall have a round timer. at the end of the timer,
    there's a mandatory vote. since llms don't experience human-time,
    we simulate the round timer by setting a maximum count of real turns.

tools
  all players
    [ASK](player_id; question) – only usable on a real turn
    [ANSWER](response) – usable after being [ASK]ed
    [VOTE](player_id) – usable during any vote
  spy only
    [GUESS](location_name) - usable on virtual turns and during the
                              end-of-round vote

turn structure
  1. a random player is chosen for the first real turn and asks a
     question to another player using [ASK]
  2. the recipient answers using [ANSWER]
  3. virtual turn rotation occurs (all players in random order). every
     player may VOTE or GUESS (if spy). anything else is a no-op pass.
  4. if no game-ending event, the answerer becomes the new asker (goto 1).
  5. after `max_real_turns` the round ends. a forced vote is called
     (the spy can still GUESS during their virtual turn, but they aren't
     required to, they can vote like any other player). if they guess
     correctly, they win. otherwise, they lose and the non-spies win).
     if there is no majority, the spy wins.

voting
  • a vote can be initiated on any virtual turn via [VOTE](player_id).
    when this happens, ALL players are polled in random order to cast votes.
  • a strict majority (> n/2) is required.  ties count as failed votes.
  • mid-game failed votes: round continues.
  • end-of-round failed votes: spy wins.
  • during any vote, the spy may [GUESS].
"""

import random
import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from textarena.envs.Spyfall.locations import LOCATIONS as DEFAULT_LOCATIONS

# ── constants ───────────────────────────────────────────────────────────


class Phase(Enum):
    ASK = auto()  # current player must ASK someone
    ANSWER = auto()  # current player answers a question
    VIRTUAL = auto()  # virtual turn: player may VOTE/GUESS or no-op
    VOTE = auto()  # collecting votes from all players


ASK_PATTERN = re.compile(r"\[ASK\]\s*\((\d+)\s*;\s*(.+?)\)", re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(r"\[ANSWER\]\s*\((.+?)\)", re.IGNORECASE)
VOTE_PATTERN = re.compile(r"\[VOTE\]\s*\((\d+)\)", re.IGNORECASE)
GUESS_PATTERN = re.compile(r"\[GUESS\]\s*\((.+?)\)", re.IGNORECASE)

# ── environment ───────────────────────────────────────────────────────────


class SpyfallEnv(ta.Env):
    """
    spyfall environment for TextArena.

    max_real_turns: int
        maximum number of question-answer pairs before the round ends
    locations: dict[str, list[str]] | None
        custom location→roles mapping. falls back to defaults in locations.py
    error_allowance: int
        number of consecutive invalid moves a player gets before auto-losing
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

    def reset(self, num_players: int, seed: Optional[int] = None):
        assert num_players >= 3, "spyfall requires at least 3 players."

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
            sampled = list(available_roles)
            random.shuffle(sampled)
            while len(sampled) < non_spy_count:
                sampled.append(random.choice(available_roles))

        role_idx = 0
        for pid in range(num_players):
            if pid == spy_id:
                roles[pid] = "spy"
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
                pid: ("spy" if pid == spy_id else "non-spy")
                for pid in range(num_players)
            },
        )

        self.phase = Phase.ASK
        self.next_player_ids: List[int] = []
        self._saved_virtual_queue: List[int] = []
        first_asker = random.randint(0, num_players - 1)
        self.state.manually_set_current_player_id(first_asker)

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        n = self.state.num_players
        players_list = ", ".join(f"Player {i}" for i in range(n))
        location_list = ", ".join(sorted(self.locations.keys()))

        base = (
            f"Welcome to the game of Spyfall! You are Player {player_id}.\n"
            f"Players: {players_list}\n"
            f"Max questions per round: {self.max_real_turns}\n\n"
        )

        if game_state["roles"][player_id] == "spy":
            base += (
                "You are the SPY. You do NOT know the location.\n"
                "The other players know the location and each have an assigned role.\n"
                "If the other players win before you, you lose.\n"
                "\n"
                "You can win by:\n"
                "- Guessing the location before the other players discover that you're the spy.\n"
                "- Remaining undetected as the spy for the whole round.\n"
                "- Convincing the remaining players to accuse the wrong player as the spy.\n"
                "\n"
                "The other players can win by:\n"
                "- Accusing you to be the spy via a majority vote.\n"
                "\n"
                "Available tools — brackets, parentheses, and semicolons must appear exactly as shown:\n"
                "  [ASK](player_id; question) – e.g. [ASK](2; What's the dress code?)\n"
                "  [ANSWER](response) – e.g. [ANSWER](The dress code is casual, you see lots of baseball caps.)\n"
                "  [VOTE](player_id) – e.g. [VOTE](3)\n"
                "  [GUESS](location_name) – e.g. [GUESS](The Beach)\n"
                "  (ask: your asking turn only | answer: only after being asked\n"
                "  | vote: during votes | guess: any time)\n"
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
                "If the spy wins, you and your team lose.\n"
                "\n"
                "You can win by:\n"
                "- Identifying the spy and accusing them via a majority vote.\n"
                "\n"
                "The spy can win by:\n"
                "- Guessing the location before you discover who they are.\n"
                "- Remaining undetected for the whole round.\n"
                "- The non-spies accusing the wrong player via a majority vote.\n"
                "\n"
                "Available tools — brackets, parentheses, and semicolons must appear exactly as shown:\n"
                "  [ASK](player_id; question) – e.g. [ASK](2; What's the dress code?)\n"
                "  [ANSWER](response) – e.g. [ANSWER](The dress code is casual, you see lots of baseball caps.)\n"
                "  [VOTE](player_id) – e.g. [VOTE](3)\n"
                "  (ask: your asking turn only | answer: only after being asked\n"
                "  | vote: during votes)\n"
                "\n"
                f"Possible locations: {location_list}\n"
            )

        base += (
            "\nTurn structure:\n"
            "  1. The current asker MUST use [ASK](player_id; question) to question\n"
            "     another player. Passing is not allowed on your asking turn.\n"
            "  2. The recipient answers using [ANSWER]. Do NOT use [ASK], [VOTE],\n"
            "     or [GUESS] in your answer.\n"
            "  3. After each Q&A, every player gets a virtual turn where they\n"
            "     may [VOTE] to accuse someone (or [GUESS] if spy). Anything\n"
            "     else you type is treated as a pass.\n"
            "  4. If a vote is called, ALL players must cast a [VOTE]. The spy\n"
            "     may [GUESS] instead of voting.\n"
            "  5. The answerer becomes the next asker.\n"
            "  6. After all questions are exhausted, a final vote is called.\n"
        )
        return base

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
        self.state.add_observation(
            to_id=target,
            message=(
                f"Player {pid} is asking you: {question}. "
                "Reply using the [ANSWER] tool — do NOT use [ASK], [VOTE], or [GUESS] in your answer."
            ),
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )
        self.state.game_state["answerer_id"] = target

    def _handle_answer(self, pid: int, action: str):
        m = ANSWER_PATTERN.search(action)
        if not m:
            self._invalid(
                pid,
                (
                    "You must answer using: [ANSWER](response).\n"
                    "Example: [ANSWER](The dress code here is formal.)\n"
                ),
            )
            return

        response = m.group(1).strip()

        self.state.add_observation(
            from_id=pid,
            message=f"Player {pid} answers: {response}",
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )
        self.state.game_state["real_turn_count"] += 1
        self.state.add_observation(
            message="Answer received. Moving into virtual turn rotation — each player may now vote or pass.",
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )

    def _handle_virtual(self, pid: int, action: str):
        gs = self.state.game_state

        guess_m = GUESS_PATTERN.search(action)
        if guess_m:
            if pid != gs["spy_id"]:
                self._invalid(pid, "Only the spy can use [GUESS].")
                return
            self._resolve_guess(pid, guess_m.group(1).strip())
            return

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
            self._invalid(pid, "You MUST vote using [VOTE](player_id). Do not use [ASK] or [ANSWER] in your response.")
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

    def _resolve_vote(self):
        gs = self.state.game_state
        votes = gs["votes"]
        spy_id = gs["spy_id"]
        forced = gs["is_forced_vote"]
        n = self.state.num_players

        counts: Dict[int, int] = {}
        for target in votes.values():
            counts[target] = counts.get(target, 0) + 1

        vote_summary = ", ".join(
            f"Player {t}: {counts[t]} vote(s)" for t in sorted(counts.keys())
        )
        self.state.add_observation(
            message=f"Vote results — {vote_summary}",
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )

        # majority = strictly more than half
        # a tie or no majority is a failed vote
        majority_threshold = n // 2 + 1
        max_count = max(counts.values()) if counts else 0
        top_targets = [t for t, c in counts.items() if c == max_count]

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

        # if we have majority on a single target
        accused = top_targets[0]
        if accused == spy_id:
            self._nonspies_win(
                f"Player {accused} was correctly identified as the spy! Non-spies win!"
            )
        else:
            self._spy_wins(
                f"Player {accused} was accused, but they are NOT the spy! The spy wins!"
            )

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

    def _advance(self):
        """overarching phase management function."""
        if self.state.made_invalid_move:
            return
        if self.state.done:
            return

        gs = self.state.game_state

        # queue all *other* players after a vote is initiated
        if self.phase == Phase.VIRTUAL and gs["votes"]:
            self._save_virtual_queue()
            self._start_vote_collection()
            return

        # rotate while there are still queued players
        if self.next_player_ids:
            self.state.manually_set_current_player_id(self.next_player_ids.pop())
            return

        # queue exhausted, resolve current phase and transition

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

            # return to q&a
            self.phase = Phase.ASK
            next_asker = gs["answerer_id"]
            self.state.add_observation(
                to_id=next_asker,
                message=(
                    "It is your turn to ask. You MUST use [ASK](player_id; question) "
                    "— passing is not allowed. Example: [ASK](2; What's the dress code?)"
                ),
                observation_type=ta.ObservationType.GAME_MESSAGE,
            )
            self.state.manually_set_current_player_id(next_asker)
            return

        if self.phase == Phase.VOTE:
            self._resolve_vote()
            if self.state.done:
                return

            self._restore_after_vote()
            return

    def _start_virtual_rotation(self):
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
        self._saved_virtual_queue = list(self.next_player_ids)
        self.next_player_ids = []

    def _start_vote_collection(self):
        self.phase = Phase.VOTE
        gs = self.state.game_state
        already_voted = set(gs["votes"].keys())  # don't requeue player who called vote
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
        """after a failed mid-game vote, resume virtual turns or next q&a."""
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
        next_asker = gs["answerer_id"]
        self.state.add_observation(
            to_id=next_asker,
            message=(
                "It is your turn to ask. You MUST use [ASK](player_id; question) "
                "— passing is not allowed. Example: [ASK](2; What's the dress code?)"
            ),
            observation_type=ta.ObservationType.GAME_MESSAGE,
        )
        self.state.manually_set_current_player_id(next_asker)

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
