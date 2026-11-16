# Task 1 - Normative Specification for Eregion Works

## 1. Scope and formal vocabulary

The normative system applies to every agent `a`, active task `k`, office `r`,
and simulation step `t`. It combines achievement obligations (an outcome must
be reached before a deadline), maintenance obligations (a condition must be
kept true), prohibitions, and permissions.

The notation `O_a(phi | c)` means that agent `a` is obliged to make `phi` true
when condition `c` holds. `O_G` is a joint obligation on a group, `F_a` is a
prohibition, and `P_a` is a permission. A norm becomes active when its condition
holds and expires when it is satisfied, its deadline passes, or its underlying
task ceases to exist.

The following observable predicates connect the specification to the starter
code:

- `assigned(a,k)`: `state.agents[a].task_id == k.id`.
- `ready(a,k)`: `k.can_progress(a)` and `a` has not completed its portion.
- `effective(a,k,r)`: `assigned(a,k)`, `r == k.room_id`, the room task type
  matches `k.task_type`, and `ready(a,k)`.
- `C(r)`, `Cmax(r)`, and `theta(r)`: the room's `cleanliness`,
  `initial_cleanliness`, and `accessible_threshold`.
- `d` and `c`: `task_damage` and `clean_increase` from the scenario.
- `W(k,t)`: the minimum number of further work steps required under proper
  cooperation. It is the remaining work of the individual participant, the
  maximum participant remainder for a simultaneous task, and the sum of the
  phase remainders for a sequential task.
- `slack(k,t) = k.remaining - W(k,t)`. Smaller slack means greater urgency.
- For a proposed joint action `J`, `workers(r,J)` and `cleaners(r,J)` count work
  and clean actions directed at `r`.
- `safe(r,J)` holds when

  `C(r) - d * workers(r,J) + c * cleaners(r,J) >= theta(r)`.

This last predicate is evaluated for the whole joint step because work and
cleaning effects are combined by the environment before cleanliness is
clamped. It prevents an accepted action profile from closing an office.

## 2. Norms

### N1 - Fulfil assigned work

**Formal rule.**

`O_a(complete(a,k) before k expires | assigned(a,k))`.

At a step where `ready(a,k)` holds, work is available, and no higher-priority
maintenance or cooperation norm requires another action, this achievement
obligation induces the action obligation `O_a(Work(k.room_id))`. When several
tasks compete for help or maintenance, the task with the least slack has
priority.

**Parties.** The task participant is the bearer; the company and any task
partner are beneficiaries.

**Satisfaction and violation.** The norm is satisfied when the participant's
portion reaches `work_required` and the shared task completes before expiry. It
is violated if the task expires with that portion incomplete. An agent also
commits an omission violation when it idles despite being ready, able to work,
and not pre-empted by a higher-priority norm.

**Justification.** Completing production tasks is the agents' primary duty to
the company. The exception for higher-priority norms prevents a short-term task
gain from destroying shared productive capacity or breaking a cooperative
commitment.

### N2 - Cooperate faithfully

**Simultaneous tasks.** For participants `G = participants(k)`, when every
unfinished participant is ready and the joint work step is safe:

`O_G(and_(a in G) Work_a(k.room_id) | mode(k) = simultaneous)`.

No participant is permitted to take unilateral work intended to benefit from a
partner who does not contribute in that step. If maintenance is required, the
participants must first follow the common maintenance plan and resume work
together.

**Sequential tasks.** If `l = k.leader`, then:

- `O_l(Work(k.room_id) | ready(l,k) and safe)`;
- `F_f(Work(k.room_id) | not complete(l,k))` for every follower `f`; and
- after handoff, `O_f(Work(k.room_id) | ready(f,k) and safe)`.

**Parties.** All participants are bearers and reciprocal beneficiaries; the
company is also a beneficiary.

**Satisfaction and violation.** A simultaneous instance is satisfied by aligned
participation until every portion completes. Unilateral work, avoidable
abandonment, or deliberately leaving a partner unable to finish is a violation.
A sequential instance is satisfied when the leader completes, handoff occurs,
and the follower then completes. Out-of-order work or avoidable failure by the
active participant is a violation.

**Justification.** Cooperative tasks create reliance. The norm prevents
free-riding, wasted partner effort, and incorrect ordering. It also gives a
clear interpretation to the `leader`, `participants`, `progress`, and `mode`
fields in an observation.

### N3 - Preserve office accessibility

**Formal rule.** For the agents `G_r` whose current obligations concern room
`r`:

`O_G_r(safe(r,J) | r is accessible and productive action is planned)`.

Equivalently, `F_G_r(J | not safe(r,J))`: agents must not choose a combination
of actions that predictably takes cleanliness below the accessible threshold.
If work alone would be unsafe and sufficient cleaning is available, enough
agents must clean before, or in the same joint step as, further work. Cleaning
need not continue once the room is at its initial cleanliness and the next
planned work profile is safe.

**Parties.** Agents who use the room bear the joint duty. Every present and
future user, as well as the company, benefits from it.

**Satisfaction and violation.** A live instance is satisfied at a step when the
resulting room remains accessible. A preventable `locked` event is a violation
by agents whose work created the unsafe profile and by a designated maintainer
who omitted available cleaning. Closure caused after no compliant joint action
was physically possible is recorded as an unavoidable failure rather than an
agent violation.

**Justification.** Closure lasts many steps, ejects occupants, blocks work and
cleaning, and harms agents who did not cause it. This makes cleanliness a shared
and exhaustible resource rather than a private concern.

### N4 - Use shared resources purposefully

**Formal rules.**

- `F_a(Work(r) | not effective(a,k,r) for every active k)`.
- `F_a(Clean(r) | C(r) = Cmax(r) and no same-step work requires an offset)`.
- `P_a(Idle | no active work, cooperation, or allocated maintenance duty)`.

**Parties.** Every agent is a bearer; other agents and the company are
beneficiaries.

**Satisfaction and violation.** The norm is satisfied when accepted work makes
useful progress and cleaning either restores cleanliness or offsets joint work.
Accepted work with `Outcome.progress == 0` is evidence of a violation unless it
was made ineffective by an unforeseeable joint transition. Unnecessary cleaning
at maximum cleanliness is also a violation. Idling is compliant only when the
agent has no actionable duty, for example a sequential follower waiting for
handoff with no allocated maintenance work.

**Justification.** The simulator charges work and cleaning costs. More
seriously, ineffective work still consumes cleanliness. This norm prevents
resource damage and action costs that produce no company or social benefit.

### N5 - Share maintenance fairly

**Formal rule.** When N3 requires `q` cleaners, exactly `q` available agents are
allocated a temporary maintenance obligation:

`O_a(Clean(r) | maintainer(a,r,t))`.

Maintainers are selected by increasing opportunity cost: an unassigned agent
first, then a sequential follower waiting for handoff, then the assigned agent
with the greatest task slack. Ties rotate among equally eligible agents using a
deterministic order. Non-maintainers are permitted to work only if the complete
joint profile remains safe.

**Parties.** The selected maintainer is the immediate bearer. Equally situated
agents are reciprocal beneficiaries and future bearers.

**Satisfaction and violation.** A selected agent satisfies the instance by
performing the allocated clean action. It violates the instance by idling or
working when cleaning was available and required. Over an episode, persistent
cleaning imbalance among agents with comparable opportunities is evidence of
free-riding and therefore a fairness violation.

**Justification.** A collective duty alone allows every agent to wait for
someone else. Allocation makes responsibility action-guiding and measurable,
while opportunity-cost ordering reduces interference with urgent task work.

## 3. Conflict detection and resolution

Norms are applied using the following strict priority order:

1. **Physical feasibility:** choose only an action in `available_actions`.
2. **Correct ordering and purposeful use (N2/N4):** never perform premature
   sequential work or knowingly ineffective work.
3. **Office safety (N3):** reject any avoidably unsafe joint profile.
4. **Cooperative reliance (N2):** preserve simultaneous alignment and active
   sequential commitments.
5. **Task completion (N1):** among compatible work duties, serve the task with
   the least slack.
6. **Fair allocation (N5):** minimise and rotate the maintenance burden; idle
   only when no higher rule applies.

This ordering resolves the main conflict between working and cleaning. If the
planned work profile is safe, N1 requires work. If it is unsafe, N3 activates
and N5 chooses the cleaner or cleaners. For a simultaneous task, participants
must not break coordination: a free agent should clean if one is available;
otherwise the partners take a coordinated maintenance step before resuming
joint work. In a sequential task, the waiting follower is the preferred
maintainer, allowing the leader to work when their combined actions are safe.

If a task deadline and room safety cannot both be satisfied, preventing office
closure has priority. This is proportionate in the supplied scenario because a
closure disables the shared office for 48 steps, while the affected task is one
of many tasks in an episode. The missed task remains a recorded N1 violation,
but it is excused as a conflict-induced violation when every action that could
meet the deadline would violate N3. This distinction avoids pretending that all
obligations are simultaneously satisfiable.

When several rooms require maintenance, agents choose the room with the
smallest safety margin `C(r) - theta(r)`; remaining ties are broken by the most
urgent task in the room and then room ID. Deterministic tie-breaking lets
independent agents compute the same allocation from a shared observation.

## 4. Operational interpretation for Task 2

The rule-based policy can implement the norms in this order:

1. Remove unavailable and knowingly ineffective work actions.
2. Determine the active phase and partner readiness for the assigned task.
3. Predict whether the relevant joint work profile is safe.
4. If maintenance is required, compute the N5 maintainer allocation and clean
   when selected.
5. Otherwise work when ready, respecting simultaneous alignment and sequential
   order.
6. If unassigned, clean the highest-risk room only when allocated; otherwise
   idle.

Norm compliance can be evaluated from `StepResult` records. Suitable measures
are the proportion of ready and safe opportunities that resulted in useful
work (N1), aligned cooperative work and correct handoffs (N2), preventable room
locks (N3), accepted zero-progress work and unnecessary cleaning (N4), and the
distribution of required clean actions among eligible agents (N5).

## 5. Starter-code semantic notes

These notes should inform the implementation and experimental-method section;
they are not claims about the intended normative model.

1. The coursework text says simultaneous progress requires both agents to work
   in the same step. In the current `transitions.py` and `tasks.py`, each
   participant's progress is credited independently; there is no same-step
   check. N2 follows the stated coursework semantics and can enforce alignment
   at policy level.
2. The coursework text says an office closure loses unfinished task progress.
   The current `_settle_rooms()` implementation locks and restores the room but
   does not reset associated task progress.
3. The coursework text describes one deadline for the whole sequential task.
   The current `settle_tasks()` resets `remaining` to the full deadline at
   leader handoff, effectively giving each phase a separate time budget.
4. `Observation.task_actions` binds work and clean to the assigned task room (or
   the current room when unassigned), while `available_actions` contains actions
   for every accessible room. Cross-room maintenance therefore requires using
   `available_actions` rather than only `task_actions`.

The `src/` files should remain unchanged, as required by the coursework. Results
should be reported against the supplied executable implementation, with these
differences stated where they affect interpretation.
