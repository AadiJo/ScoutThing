Grade team {{team}} ({{alliance}} alliance) in this match.

# Match

Roster: {{roster}}
Auto winner (inactive in shifts 1 and 3): {{auto_winner}}

Phases in video seconds, with the hubs active in each:
{{phases}}

The target's own hub is active in: {{own_active}}

# Driving ledger

One call per line. `lost_s` is what a charged line costs.

{{driving_ledger}}

# Readiness ledger

One call per line. `late_s` is what a charged line costs.

{{readiness_ledger}}

# Defense items

Every fact id below must appear in an episode's `facts`.

{{defense_items}}

# Facts

Read `notes` first. Every fact has an `id`, its video time span (`t_start`, `t_end`), the
scoreboard `clock` at its start and its match `phase`.

```json
{{facts}}
```

Open the frames you need, then return the answer.
