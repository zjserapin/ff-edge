"""ff-edge — fantasy football draft research for the Shiva Bowl.

Two layers, deliberately separated:

  data      config, cache, nflverse, sleeper, adp, ids — everything that talks to
            a network and everything that normalizes what comes back
  analysis  scoring, landscape, features, archetypes, breakout, rookies,
            simulate — everything that turns that into an answer

Nothing in the analysis layer touches a network directly; it all goes through
the data layer's cache, so the same call is free the second time you make it.
"""
