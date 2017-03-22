# Thor
A library for Bayesian optimization.

## Design Schematic

Flask web server and backend database. To start, just use a simple SQLite database, but later we can expand to Postgres. 

The web server will be able to provide access to experiments, recommendations made so far, corresponding observations.

We should have support for parallel optimization, deep Bayesian optimization, integrating out length scales or maximizing the marginal likelihood. Basically, we should have the option to control a great number of behaviors of Bayesian optimization, but we should also have defaults that abstract this control away, in the event that the user does not wish to deal with it.

We will want to be able to:

1. Create an experiment.
2. Receive a recommendation.
3. While evaluating a current recommendation, formulate other recommendations.
4. Mark recommendations as pending versus completed.
5. Define boundaries for parameters (and scale these to the unit hypercube, respectively). One should additionally be able to name hyperparameters.
6. Should we develop a kind of account-based system where users have associated experiments and so on?

Finally, and this is very important, it would be great to be able to build a GUI for individual experiments.

## What do we need to store on the server side of the application?

1. We need a user identifier. Also a password.
2. Experiments need a name and identifier. We need a way to store the search space of the problem. This could be done by storing a JSON string on the server, for instance.
3. A mapping of inputs for a particular experiment to the output of the objective function of that input. Notice that the output of the objective function can be pending! This is going to enable persistence across sessions.

