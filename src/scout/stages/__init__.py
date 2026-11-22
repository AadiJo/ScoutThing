"""One module per pipeline stage. Each defines `STAGE`; `scout.pipeline` lists them in order.
A stage module may import scout.schemas, scout.run, scout.video, scout.assets and scout.games,
never another stage."""
