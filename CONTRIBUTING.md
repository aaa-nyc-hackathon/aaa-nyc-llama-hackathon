# Contributing

Basic tenets

1. All pull requests be reviewed by one member of the team prior to merging

2. While we generally require passing test harness for merging to main, in 
some circumstances we may merge anyway if the change is deemed urgent.

3. We generally expect new code to add tests that are associated with the 
added code.

4. We may add, remove, or modify tenets as deemed by the team. If there is 
a reason why you disagree with a tenet, let's write it out on an issue. 

# build

... TODO

# Testing

The github action(s) run all tests on the remote.
To run them locally, navigate to the top level directory and run

```bash
pytest --durations=7 
```

Besides running the collection and execution processes, this will display 
the 7 slowest tests so we can triage short term improvements with the biggest
gains.
