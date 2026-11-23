Grade team {{team}} ({{alliance}} alliance) in this match.

# Match

Roster: {{roster}}
Auto winner (inactive in shifts 1 and 3): {{auto_winner}}

Phases in video seconds, with the hubs active in each:
{{phases}}

The target's own hub is active in: {{own_active}}
The corner camera panel showing its own end: {{home_view}}

# What was seen

Code computed these from the facts. Write nothing about untracked time as if it were seen.

{{coverage}}

# Driving ledger

Each barrier line's DEFAULT stands unless you overrule it on a tile (see the rules); a charged
line costs its `lost_s`. Each `loaded_trip` line needs a hopper call from its HOPPER TILES.

{{driving_ledger}}

# Readiness ledger

Each DEFAULT stands unless you overrule it on a tile; a charged line costs `late_s`. A
`not_judged` line costs nothing and is not scored unless a tile settles it (`home` or
`not_home`).

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
